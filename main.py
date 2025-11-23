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
from typing import Optional

import qrcode
import pyotp
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from PIL import Image, ImageDraw, ImageFont
from threading import Lock
import copy
import yaml

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUTBOX_DIR = ROOT / "outbox"
STATIC_DIR = ROOT / "static"
STORE_FILE = DATA_DIR / "vouchers.yaml"
STORE_LOCK = Lock()

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


class VoucherStatus(str, enum.Enum):
    active = "active"
    redeemed = "redeemed"


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


def voucher_card_image(voucher: VoucherOut) -> bytes:
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


def card_base64_for(voucher: VoucherOut) -> str:
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


def require_admin_session(request: Request) -> dict:
    session = get_session_by_token(request.cookies.get(SESSION_COOKIE))
    if not session:
        raise HTTPException(status_code=401, detail="Login erforderlich")
    return session


def send_email_stub(voucher: VoucherOut, voucher_png: bytes) -> None:
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


def serialize_datetime(value: Optional[dt.datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def parse_datetime(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    return dt.datetime.fromisoformat(value)


def _default_store() -> dict:
    return {
        "next_id": 1,
        "vouchers": [],
        "feature_flag": None,
        "sessions": [],
    }


def _load_store() -> dict:
    if not STORE_FILE.exists():
        return _default_store()
    with open(STORE_FILE, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if "next_id" not in data:
        data["next_id"] = 1
    data.setdefault("vouchers", [])
    data.setdefault("sessions", [])
    if data.get("feature_flag") is None:
        data["feature_flag"] = None
    return data


def _save_store(data: dict) -> None:
    STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STORE_FILE, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)


def _read_store() -> dict:
    with STORE_LOCK:
        return copy.deepcopy(_load_store())


def _mutate_store(mutator):
    with STORE_LOCK:
        data = _load_store()
        result = mutator(data)
        _save_store(data)
        return result


def list_voucher_records() -> list[dict]:
    return _read_store().get("vouchers", [])


def get_voucher_record(code: str) -> Optional[dict]:
    for record in list_voucher_records():
        if record["code"] == code:
            return record
    return None


def upsert_voucher_record(record: dict) -> dict:
    def mutator(data: dict):
        for idx, existing in enumerate(data["vouchers"]):
            if existing["code"] == record["code"]:
                data["vouchers"][idx] = record
                break
        else:
            data["vouchers"].append(record)
        return record

    return _mutate_store(mutator)


def delete_voucher_record(code: str) -> Optional[dict]:
    def mutator(data: dict):
        for idx, existing in enumerate(data["vouchers"]):
            if existing["code"] == code:
                removed = data["vouchers"].pop(idx)
                return removed
        return None

    return _mutate_store(mutator)


def record_to_out(record: dict) -> VoucherOut:
    base = VoucherOut(
        id=record["id"],
        code=record["code"],
        status=VoucherStatus(record["status"]),
        amount=amount_from_cents(record.get("amount")),
        email=record.get("email"),
        recipient_name=record.get("recipient_name"),
        note=record.get("note"),
        created_at=parse_datetime(record.get("created_at")),
        redeemed_at=parse_datetime(record.get("redeemed_at")),
        redeem_hint=f"Scan QR oder POST /vouchers/{record['code']}/redeem",
        qr_base64="",
        card_base64=None,
    )
    qr = qr_base64_for(record["code"])
    card = card_base64_for(base)
    return base.model_copy(update={"qr_base64": qr, "card_base64": card})


def create_voucher_record(payload: VoucherCreate, *, device_id: Optional[str] = None, code: Optional[str] = None) -> dict:
    def mutator(data: dict):
        now = dt.datetime.utcnow()
        record = {
            "id": data.get("next_id", 1),
            "code": code or uuid.uuid4().hex,
            "email": payload.email,
            "recipient_name": payload.recipient_name,
            "amount": amount_to_cents(payload.amount),
            "note": payload.note,
            "delivery_method": payload.delivery_method.value,
            "device_id": device_id,
            "status": VoucherStatus.active.value,
            "created_at": now.isoformat(),
            "redeemed_at": None,
        }
        data["next_id"] = record["id"] + 1
        data.setdefault("vouchers", []).append(record)
        return copy.deepcopy(record)

    return _mutate_store(mutator)


def update_voucher_record(record: dict) -> dict:
    return upsert_voucher_record(record)


def list_vouchers_out() -> list[VoucherOut]:
    return [record_to_out(r) for r in list_voucher_records()]


def find_voucher_out(code: str) -> VoucherOut:
    record = get_voucher_record(code)
    if not record:
        raise HTTPException(status_code=404, detail="Voucher not found")
    return record_to_out(record)


def latest_device_voucher(device_id: str) -> Optional[dict]:
    if not device_id:
        return None
    records = [r for r in list_voucher_records() if r.get("device_id") == device_id]
    if not records:
        return None
    records.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return records[0]


def creation_enabled() -> AdminStatus:
    data = _read_store()
    flag = data.get("feature_flag")
    if not flag or not flag.get("expires_at"):
        return AdminStatus(active=False, enabled_until=None)
    expires_at = parse_datetime(flag.get("expires_at"))
    active = expires_at is not None and expires_at > dt.datetime.utcnow()
    return AdminStatus(active=active, enabled_until=expires_at)


def set_creation_window(minutes: int = CREATION_WINDOW_MINUTES) -> AdminStatus:
    expires_at = dt.datetime.utcnow() + dt.timedelta(minutes=minutes)

    def mutator(data: dict):
        data["feature_flag"] = {"expires_at": expires_at.isoformat()}
        return data["feature_flag"]

    _mutate_store(mutator)
    return AdminStatus(active=True, enabled_until=expires_at)


def require_creation_enabled() -> None:
    status = creation_enabled()
    if not status.active:
        raise HTTPException(status_code=403, detail="Voucher-Erstellung derzeit nicht freigeschaltet.")


def create_admin_session() -> dict:
    def mutator(data: dict):
        now = dt.datetime.utcnow()
        expires_at = now + dt.timedelta(hours=SESSION_DURATION_HOURS)
        token = secrets.token_urlsafe(32)
        sessions = []
        for entry in data.get("sessions", []):
            exp = parse_datetime(entry.get("expires_at"))
            if exp and exp > now:
                sessions.append({"token": entry["token"], "expires_at": exp.isoformat()})
        sessions.append({"token": token, "expires_at": expires_at.isoformat()})
        data["sessions"] = sessions
        return {"token": token, "expires_at": expires_at}

    return _mutate_store(mutator)


def get_session_by_token(token: Optional[str]) -> Optional[dict]:
    if not token:
        return None

    def mutator(data: dict):
        now = dt.datetime.utcnow()
        found = None
        sessions = []
        for entry in data.get("sessions", []):
            exp = parse_datetime(entry.get("expires_at"))
            if not exp or exp <= now:
                continue
            if entry["token"] == token:
                found = {"token": token, "expires_at": exp}
            sessions.append({"token": entry["token"], "expires_at": exp.isoformat()})
        data["sessions"] = sessions
        return found

    return _mutate_store(mutator)


ensure_folders()

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
    admin_session: dict = Depends(require_admin_session),
) -> VoucherOut:
    require_creation_enabled()
    record = create_voucher_record(payload)
    voucher = record_to_out(record)
    if payload.delivery_method == VoucherDelivery.email and payload.email:
        voucher_png = generate_qr(signed_payload(voucher.code))
        send_email_stub(voucher, voucher_png)
    return voucher


@app.get("/vouchers", response_model=list[VoucherOut])
def list_vouchers(
    admin_session: dict = Depends(require_admin_session),
) -> list[VoucherOut]:
    return list_vouchers_out()


@app.get("/vouchers/{code}", response_model=VoucherOut)
def get_voucher(
    code: str,
    admin_session: dict = Depends(require_admin_session),
) -> VoucherOut:
    return find_voucher_out(code)


@app.post("/vouchers/{code}/send_email", response_model=VoucherOut)
def send_voucher_email(code: str, payload: EmailSendPayload) -> VoucherOut:
    record = get_voucher_record(code)
    if not record:
        raise HTTPException(status_code=404, detail="Voucher not found")
    record["email"] = payload.email
    update_voucher_record(record)
    voucher = record_to_out(record)
    voucher_png = generate_qr(signed_payload(voucher.code))
    send_email_stub(voucher, voucher_png)
    return voucher


@app.get("/vouchers/{code}/qr.png")
def voucher_qr_png(code: str) -> StreamingResponse:
    record = get_voucher_record(code)
    if not record:
        raise HTTPException(status_code=404, detail="Voucher not found")
    img_bytes = generate_qr(signed_payload(record["code"]))
    return StreamingResponse(BytesIO(img_bytes), media_type="image/png")


@app.get("/vouchers/{code}/card.png")
def voucher_card_png(
    code: str,
    admin_session: dict = Depends(require_admin_session),
) -> StreamingResponse:
    record = get_voucher_record(code)
    if not record:
        raise HTTPException(status_code=404, detail="Voucher not found")
    voucher = record_to_out(record)
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
) -> AdminStatus:
    validate_admin(payload)
    session = create_admin_session()
    response.set_cookie(
        SESSION_COOKIE,
        session["token"],
        max_age=SESSION_DURATION_HOURS * 3600,
        httponly=True,
        samesite="Lax",
    )
    return creation_enabled()


@app.get("/admin/status", response_model=AdminStatus)
def admin_status(
    admin_session: dict = Depends(require_admin_session),
) -> AdminStatus:
    return creation_enabled()


@app.post("/admin/enable", response_model=AdminStatus)
def admin_enable(
    admin_session: dict = Depends(require_admin_session),
) -> AdminStatus:
    return set_creation_window()


@app.post("/vouchers/{code}/redeem", response_model=RedeemResult)
def redeem_voucher(
    code: str,
    signature: Optional[str] = Query(None),
) -> RedeemResult:
    if signature is not None:
        validate_signature(code, signature)
    record = get_voucher_record(code)
    if not record:
        raise HTTPException(status_code=404, detail="Voucher not found")
    if record["status"] == VoucherStatus.redeemed.value:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Voucher already redeemed",
                "redeemed_at": record.get("redeemed_at"),
            },
        )

    record["status"] = VoucherStatus.redeemed.value
    record["redeemed_at"] = dt.datetime.utcnow().isoformat()
    update_voucher_record(record)

    return RedeemResult(
        code=record["code"],
        status=VoucherStatus.redeemed,
        redeemed_at=parse_datetime(record["redeemed_at"]),
    )


@app.delete("/vouchers/{code}", response_model=RedeemResult)
def delete_voucher(
    code: str,
    admin_session: dict = Depends(require_admin_session),
) -> RedeemResult:
    record = delete_voucher_record(code)
    if not record:
        raise HTTPException(status_code=404, detail="Voucher not found")
    return RedeemResult(code=code, status=VoucherStatus.redeemed, redeemed_at=parse_datetime(record.get("redeemed_at")))


@app.post("/claim", response_model=VoucherOut)
def claim_voucher(request: Request, response: Response) -> VoucherOut:
    device_id = request.cookies.get(DEVICE_COOKIE) or uuid.uuid4().hex
    record = latest_device_voucher(device_id)
    if record:
        voucher = record_to_out(record)
    else:
        require_creation_enabled()
        payload = VoucherCreate()
        new_record = create_voucher_record(payload, device_id=device_id, code=uuid.uuid4().hex)
        voucher = record_to_out(new_record)

    response.set_cookie(
        key=DEVICE_COOKIE,
        value=device_id,
        max_age=60 * 60 * 24 * 365,
        httponly=False,
        samesite="Lax",
    )
    return voucher
