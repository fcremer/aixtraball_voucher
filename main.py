import base64
import datetime as dt
import enum
import hmac
import hashlib
import os
import uuid
import secrets
from io import BytesIO
from pathlib import Path
from typing import Generator, Optional

import qrcode
import pyotp
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import Column, DateTime, Enum as SqlEnum, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUTBOX_DIR = ROOT / "outbox"
STATIC_DIR = ROOT / "static"

DATABASE_URL = f"sqlite:///{DATA_DIR / 'vouchers.db'}"
DEVICE_COOKIE = "voucher_device_id"
BRAND_PRIMARY = (210, 55, 41)  # #D23729
BRAND_DARK = (165, 35, 23)  # #A52317
CARD_BG = "#f7f7f9"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "aixtraball-aixtraball210!")
ADMIN_TOTP_SECRET = os.getenv("ADMIN_TOTP_SECRET", "M2CVTQVEKUS64W6L2Z2RC6AUWIWGUXDJ")  # Base32, default for testing
CREATION_FLAG_KEY = "voucher_creation"
CREATION_WINDOW_MINUTES = 60
SESSION_COOKIE = "admin_session"
SESSION_DURATION_HOURS = 8
VOUCHER_SIGNING_SECRET = os.getenv("VOUCHER_SIGNING_SECRET", "CHANGE_ME_SIGNING_SECRET")

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class VoucherStatus(str, enum.Enum):
    active = "active"
    redeemed = "redeemed"


class Voucher(Base):
    __tablename__ = "vouchers"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, nullable=True)
    recipient_name = Column(String, nullable=True)
    amount = Column(Integer, nullable=True)  # store cents to avoid float issues
    note = Column(Text, nullable=True)
    delivery_method = Column(String, default="download")
    device_id = Column(String, nullable=True, index=True)
    status = Column(SqlEnum(VoucherStatus), default=VoucherStatus.active, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    redeemed_at = Column(DateTime, nullable=True)


class FeatureFlag(Base):
    __tablename__ = "feature_flags"

    key = Column(String, primary_key=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)


class AdminSession(Base):
    __tablename__ = "admin_sessions"

    token = Column(String, primary_key=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)


class VoucherDelivery(str, enum.Enum):
    download = "download"
    email = "email"


class VoucherCreate(BaseModel):
    email: Optional[EmailStr] = Field(None, description="Optional email to send the voucher to.")
    recipient_name: Optional[str] = Field(None, description="Name of the person receiving the voucher.")
    amount: Optional[float] = Field(None, description="Amount of the voucher in the chosen currency.")
    note: Optional[str] = Field(None, description="Internal note for the voucher.")
    delivery_method: VoucherDelivery = Field(
        VoucherDelivery.download,
        description="How the voucher should be delivered. Email writes a stub email into /outbox.",
    )


class VoucherOut(BaseModel):
    id: int
    code: str
    status: VoucherStatus
    redeem_hint: str
    qr_base64: str
    card_base64: Optional[str]
    amount: Optional[float]
    email: Optional[EmailStr]
    recipient_name: Optional[str]
    note: Optional[str]
    created_at: dt.datetime
    redeemed_at: Optional[dt.datetime]

    class Config:
        from_attributes = True


class RedeemResult(BaseModel):
    code: str
    status: VoucherStatus
    redeemed_at: Optional[dt.datetime]


class EmailSendPayload(BaseModel):
    email: EmailStr


class AdminLogin(BaseModel):
    password: str
    otp: str


class AdminStatus(BaseModel):
    active: bool
    enabled_until: Optional[dt.datetime]


def ensure_folders() -> None:
    for path in (DATA_DIR, OUTBOX_DIR, STATIC_DIR):
        path.mkdir(parents=True, exist_ok=True)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(font_name, size)
    except Exception:
        return ImageFont.load_default()


def qr_image(content: str, size: int = 340) -> Image.Image:
    base_qr = qrcode.make(content, box_size=10, border=2).convert("RGB")
    if size and base_qr.size != (size, size):
        base_qr = base_qr.resize((size, size), Image.LANCZOS)
    return base_qr


def generate_qr(content: str) -> bytes:
    img = qr_image(content)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def voucher_card_image(voucher: Voucher) -> bytes:
    width, height = 1200, 640
    margin = 38
    card_radius = 40
    ticket = Image.new("RGB", (width, height), CARD_BG)
    draw = ImageDraw.Draw(ticket)

    # Base card
    card_rect = (margin, margin, width - margin, height - margin)
    draw.rounded_rectangle(card_rect, radius=card_radius, fill="#ffffff", outline="#e6e9ef", width=3)

    # Header
    header_rect = (margin + 18, margin + 18, width - margin - 18, margin + 170)
    draw.rounded_rectangle(header_rect, radius=28, fill=f"rgb{BRAND_PRIMARY}")
    draw.rectangle((header_rect[0], header_rect[3] - 12, header_rect[2], header_rect[3]), fill=f"rgb{BRAND_DARK}")
    draw.text((header_rect[0] + 24, header_rect[1] + 28), "Aixtraball", fill="#ffffff", font=load_font(52, bold=True))
    draw.text(
        (header_rect[0] + 24, header_rect[1] + 94),
        "Gutschein – 5€ Eintrittsrabatt",
        fill="#ffe4de",
        font=load_font(34, bold=True),
    )
    draw.text(
        (header_rect[0] + 24, header_rect[1] + 136),
        "Mehr Infos auf www.aixtraball.de",
        fill="#ffe4de",
        font=load_font(22),
    )

    # Left content
    content_x = margin + 32
    content_y = header_rect[3] + 34
    label_font = load_font(20, bold=True)
    value_font = load_font(32, bold=True)

    amount_value = "5 € Eintrittsrabatt"
    if voucher.amount is not None:
        amount_value = f"{amount_from_cents(voucher.amount):.2f} €"
    draw.text((content_x, content_y), "Gutscheinhöhe", fill="#475569", font=label_font)
    draw.text((content_x, content_y + 28), amount_value, fill=f"rgb{BRAND_PRIMARY}", font=value_font)

    note_y = content_y + 100
    draw.text((content_x, note_y), "Vor Ort vorzeigen, Entwertung durch Mitarbeitende.", fill="#111827", font=load_font(22))
    draw.text((content_x, note_y + 34), "Gutscheincode (für Team):", fill="#475569", font=label_font)
    draw.text((content_x, note_y + 60), voucher.code, fill="#111827", font=load_font(28, bold=True))

    link_y = note_y + 116
    draw.rounded_rectangle(
        (content_x, link_y, content_x + 520, link_y + 110),
        radius=18,
        fill="#fef3f2",
        outline=f"rgb{BRAND_PRIMARY}",
        width=2,
    )
    draw.text((content_x + 14, link_y + 16), "Öffnungszeiten & Infos:", fill=f"rgb{BRAND_PRIMARY}", font=label_font)
    draw.text((content_x + 14, link_y + 44), "www.aixtraball.de", fill="#7f1d1d", font=load_font(22, bold=True))

    # QR on the right
    qr_size = 300
    qr_box_w = qr_size + 64
    qr_x0 = width - margin - qr_box_w - 8
    qr_y0 = header_rect[3] + 28
    draw.rounded_rectangle(
        (qr_x0, qr_y0, qr_x0 + qr_box_w, qr_y0 + qr_size + 96),
        radius=30,
        fill="#f8fafc",
        outline="#e6e9ef",
        width=2,
    )
    qr = qr_image(signed_payload(voucher.code), size=qr_size)
    qr_with_border = Image.new("RGB", (qr_size + 16, qr_size + 16), "white")
    qr_with_border.paste(qr, (8, 8))
    ticket.paste(qr_with_border, (qr_x0 + 22, qr_y0 + 18))
    draw.text(
        (qr_x0 + 22, qr_y0 + qr_size + 36),
        "Nur Team entwertet",
        fill="#334155",
        font=load_font(22, bold=True),
    )
    draw.text(
        (qr_x0 + 22, qr_y0 + qr_size + 64),
        "QR im Entwert-Tool scannen",
        fill="#6b7280",
        font=load_font(18),
    )

    buffer = BytesIO()
    ticket.save(buffer, format="PNG")
    return buffer.getvalue()


def qr_base64_for(code: str) -> str:
    img_bytes = generate_qr(signed_payload(code))
    return base64.b64encode(img_bytes).decode("ascii")


def card_base64_for(voucher: Voucher) -> str:
    img_bytes = voucher_card_image(voucher)
    return base64.b64encode(img_bytes).decode("ascii")


def voucher_signature(code: str) -> str:
    digest = hmac.new(
        VOUCHER_SIGNING_SECRET.encode("utf-8"),
        msg=code.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def signed_payload(code: str) -> str:
    signature = voucher_signature(code)
    return f"voucher:{code}:{signature}"


def validate_signature(code: str, signature: Optional[str]) -> None:
    if signature is None:
        raise HTTPException(status_code=400, detail="Signatur erforderlich.")
    expected = voucher_signature(code)
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=400, detail="Ungültige Signatur.")


def send_email_stub(voucher: Voucher, voucher_png: bytes) -> None:
    OUTBOX_DIR.mkdir(exist_ok=True, parents=True)
    filename = OUTBOX_DIR / f"voucher_{voucher.code}.txt"
    redeem_hint = f"Redeem via QR code or POST /vouchers/{voucher.code}/redeem"
    amount_text = f"{amount_from_cents(voucher.amount):.2f} €" if voucher.amount is not None else "5 € Eintrittsrabatt"
    with open(filename, "w", encoding="utf-8") as email_file:
        email_file.write(
            f"To: {voucher.email}\n"
            f"Subject: Dein Gutschein {voucher.code}\n\n"
            f"Hallo {voucher.recipient_name or 'Voucher-Empfänger'},\n\n"
            f"hier ist dein Gutschein. Vorteil: {amount_text}.\n"
            f"Code: {voucher.code}\n"
            f"{redeem_hint}\n"
            f"Die QR-Grafik liegt als Base64 bei.\n\n"
            f"{base64.b64encode(voucher_png).decode('ascii')}\n"
        )


def amount_to_cents(amount: Optional[float]) -> Optional[int]:
    return None if amount is None else int(round(amount * 100))


def amount_from_cents(amount: Optional[int]) -> Optional[float]:
    return None if amount is None else amount / 100


def voucher_to_out(voucher: Voucher) -> VoucherOut:
    return VoucherOut(
        id=voucher.id,
        code=voucher.code,
        status=voucher.status,
        amount=amount_from_cents(voucher.amount),
        email=voucher.email,
        recipient_name=voucher.recipient_name,
        note=voucher.note,
        created_at=voucher.created_at,
        redeemed_at=voucher.redeemed_at,
        qr_base64=qr_base64_for(voucher.code),
        card_base64=card_base64_for(voucher),
        redeem_hint=f"Scan QR oder POST /vouchers/{voucher.code}/redeem",
    )


def creation_enabled(db: Session) -> AdminStatus:
    flag = db.query(FeatureFlag).filter(FeatureFlag.key == CREATION_FLAG_KEY).first()
    now = dt.datetime.utcnow()
    if not flag or not flag.expires_at:
        return AdminStatus(active=False, enabled_until=None)
    return AdminStatus(active=flag.expires_at > now, enabled_until=flag.expires_at)


def set_creation_window(db: Session, minutes: int = CREATION_WINDOW_MINUTES) -> AdminStatus:
    expires_at = dt.datetime.utcnow() + dt.timedelta(minutes=minutes)
    flag = db.query(FeatureFlag).filter(FeatureFlag.key == CREATION_FLAG_KEY).first()
    if not flag:
        flag = FeatureFlag(key=CREATION_FLAG_KEY, expires_at=expires_at)
        db.add(flag)
    else:
        flag.expires_at = expires_at
    db.commit()
    db.refresh(flag)
    return AdminStatus(active=True, enabled_until=expires_at)


def require_creation_enabled(db: Session) -> None:
    status = creation_enabled(db)
    if not status.active:
        raise HTTPException(status_code=403, detail="Voucher-Erstellung derzeit nicht freigeschaltet.")


def create_admin_session(db: Session) -> AdminSession:
    token = secrets.token_urlsafe(32)
    expires_at = dt.datetime.utcnow() + dt.timedelta(hours=SESSION_DURATION_HOURS)
    session = AdminSession(token=token, expires_at=expires_at)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session_by_token(db: Session, token: str) -> Optional[AdminSession]:
    if not token:
        return None
    session = db.query(AdminSession).filter(AdminSession.token == token).first()
    if not session:
        return None
    if session.expires_at < dt.datetime.utcnow():
        db.delete(session)
        db.commit()
        return None
    return session


def require_admin_session(request: Request, db: Session = Depends(get_db)) -> AdminSession:
    token = request.cookies.get(SESSION_COOKIE)
    session = get_session_by_token(db, token)
    if not session:
        raise HTTPException(status_code=401, detail="Login erforderlich")
    return session


def ensure_schema():
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("vouchers")]
    if "device_id" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE vouchers ADD COLUMN device_id VARCHAR"))


ensure_folders()
ensure_schema()

app = FastAPI(title="Voucher Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/vouchers", response_model=VoucherOut)
def create_voucher(
    payload: VoucherCreate,
    db: Session = Depends(get_db),
    admin_session: AdminSession = Depends(require_admin_session),
) -> VoucherOut:
    require_creation_enabled(db)
    code = uuid.uuid4().hex
    voucher = Voucher(
        code=code,
        email=payload.email,
        recipient_name=payload.recipient_name,
        amount=amount_to_cents(payload.amount),
        note=payload.note,
        delivery_method=payload.delivery_method.value,
        status=VoucherStatus.active,
    )
    db.add(voucher)
    db.commit()
    db.refresh(voucher)

    voucher_png = generate_qr(signed_payload(voucher.code))
    if payload.delivery_method == VoucherDelivery.email and payload.email:
        send_email_stub(voucher, voucher_png)

    return voucher_to_out(voucher)


@app.get("/vouchers", response_model=list[VoucherOut])
def list_vouchers(
    db: Session = Depends(get_db),
    admin_session: AdminSession = Depends(require_admin_session),
) -> list[VoucherOut]:
    vouchers = db.query(Voucher).order_by(Voucher.created_at.desc()).all()
    return [voucher_to_out(v) for v in vouchers]


@app.get("/vouchers/{code}", response_model=VoucherOut)
def get_voucher(
    code: str,
    db: Session = Depends(get_db),
    admin_session: AdminSession = Depends(require_admin_session),
) -> VoucherOut:
    voucher = db.query(Voucher).filter(Voucher.code == code).first()
    if not voucher:
        raise HTTPException(status_code=404, detail="Voucher not found")
    return voucher_to_out(voucher)


@app.post("/vouchers/{code}/send_email", response_model=VoucherOut)
def send_voucher_email(code: str, payload: EmailSendPayload, db: Session = Depends(get_db)) -> VoucherOut:
    voucher = db.query(Voucher).filter(Voucher.code == code).first()
    if not voucher:
        raise HTTPException(status_code=404, detail="Voucher not found")
    voucher.email = payload.email
    db.add(voucher)
    db.commit()
    db.refresh(voucher)

    voucher_png = generate_qr(signed_payload(voucher.code))
    send_email_stub(voucher, voucher_png)
    return voucher_to_out(voucher)


@app.get("/vouchers/{code}/qr.png")
def voucher_qr_png(code: str, db: Session = Depends(get_db)) -> StreamingResponse:
    voucher = db.query(Voucher).filter(Voucher.code == code).first()
    if not voucher:
        raise HTTPException(status_code=404, detail="Voucher not found")
    img_bytes = generate_qr(signed_payload(voucher.code))
    return StreamingResponse(BytesIO(img_bytes), media_type="image/png")


@app.get("/vouchers/{code}/card.png")
def voucher_card_png(
    code: str,
    db: Session = Depends(get_db),
    admin_session: AdminSession = Depends(require_admin_session),
) -> StreamingResponse:
    voucher = db.query(Voucher).filter(Voucher.code == code).first()
    if not voucher:
        raise HTTPException(status_code=404, detail="Voucher not found")
    img_bytes = voucher_card_image(voucher)
    return StreamingResponse(BytesIO(img_bytes), media_type="image/png")


def validate_admin(payload: AdminLogin) -> None:
    if payload.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")
    totp = pyotp.TOTP(ADMIN_TOTP_SECRET)
    if not totp.verify(payload.otp, valid_window=1):
        raise HTTPException(status_code=401, detail="Invalid OTP")


@app.post("/admin/login", response_model=AdminStatus)
def admin_login(
    payload: AdminLogin,
    response: Response,
    db: Session = Depends(get_db),
) -> AdminStatus:
    validate_admin(payload)
    session = create_admin_session(db)
    response.set_cookie(
        SESSION_COOKIE,
        session.token,
        max_age=SESSION_DURATION_HOURS * 3600,
        httponly=True,
        samesite="Lax",
    )
    return creation_enabled(db)


@app.get("/admin/status", response_model=AdminStatus)
def admin_status(
    db: Session = Depends(get_db),
    admin_session: AdminSession = Depends(require_admin_session),
) -> AdminStatus:
    return creation_enabled(db)


@app.post("/admin/enable", response_model=AdminStatus)
def admin_enable(
    db: Session = Depends(get_db),
    admin_session: AdminSession = Depends(require_admin_session),
) -> AdminStatus:
    return set_creation_window(db)


@app.post("/vouchers/{code}/redeem", response_model=RedeemResult)
def redeem_voucher(
    code: str,
    signature: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> RedeemResult:
    if signature is not None:
        validate_signature(code, signature)
    voucher = db.query(Voucher).filter(Voucher.code == code).first()
    if not voucher:
        raise HTTPException(status_code=404, detail="Voucher not found")
    if voucher.status == VoucherStatus.redeemed:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Voucher already redeemed",
                "redeemed_at": voucher.redeemed_at.isoformat() if voucher.redeemed_at else None,
            },
        )

    voucher.status = VoucherStatus.redeemed
    voucher.redeemed_at = dt.datetime.utcnow()
    db.add(voucher)
    db.commit()
    db.refresh(voucher)

    return RedeemResult(
        code=voucher.code,
        status=voucher.status,
        redeemed_at=voucher.redeemed_at,
    )


@app.delete("/vouchers/{code}", response_model=RedeemResult)
def delete_voucher(
    code: str,
    db: Session = Depends(get_db),
    admin_session: AdminSession = Depends(require_admin_session),
) -> RedeemResult:
    voucher = db.query(Voucher).filter(Voucher.code == code).first()
    if not voucher:
        raise HTTPException(status_code=404, detail="Voucher not found")
    db.delete(voucher)
    db.commit()
    return RedeemResult(code=code, status=VoucherStatus.redeemed, redeemed_at=voucher.redeemed_at)


@app.post("/claim", response_model=VoucherOut)
def claim_voucher(request: Request, response: Response, db: Session = Depends(get_db)) -> VoucherOut:
    device_id = request.cookies.get(DEVICE_COOKIE) or uuid.uuid4().hex
    existing = (
        db.query(Voucher)
        .filter(Voucher.device_id == device_id)
        .order_by(Voucher.created_at.desc())
        .first()
    )
    if existing:
        voucher = existing
    else:
        require_creation_enabled(db)
        voucher = Voucher(
            code=uuid.uuid4().hex,
            device_id=device_id,
            status=VoucherStatus.active,
            delivery_method=VoucherDelivery.download.value,
            created_at=dt.datetime.utcnow(),
        )
        db.add(voucher)
        db.commit()
        db.refresh(voucher)

    response.set_cookie(
        key=DEVICE_COOKIE,
        value=device_id,
        max_age=60 * 60 * 24 * 365,
        httponly=False,
        samesite="Lax",
    )
    return voucher_to_out(voucher)
