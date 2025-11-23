# Wallet Voucher Service

Python/FastAPI Anwendung zum Erstellen, Speichern und Entwerten von einmaligen Gutscheinen. Jeder Gutschein bekommt einen eindeutigen Code, der als QR-Grafik gespeichert/heruntergeladen oder stub-mäßig per E-Mail (Datei im Ordner `outbox/`) zugestellt werden kann. Ein Web-Frontend erlaubt das Generieren sowie das Entwerten per QR-Scan.

## Schnellstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Danach:

- Web-Frontend zum Erstellen: `http://127.0.0.1:8000/static/index.html`
- Redeem-Webseite (QR-Scanner): `http://127.0.0.1:8000/static/redeem.html`
- Endkunden-Claim-URL (per QR verteilen): `http://127.0.0.1:8000/static/claim.html` – erstellt automatisch genau einen Gutschein pro Gerät (per Cookie).
- API Healthcheck: `GET /healthz`

## API Überblick

- `POST /vouchers` – erstellt einen neuen Gutschein. Body Felder: `amount` (optional, Zahl), `recipient_name`, `email` (optional), `note`, `delivery_method` (`download` oder `email`). Antwort enthält QR als Base64 und Download-Link.
- `GET /vouchers` – Liste aller Gutscheine inkl. QR-Base64.
- `GET /vouchers/{code}` – Details zu einem Gutschein.
- `GET /vouchers/{code}/qr.png` – QR-PNG für Druck/Wallet.
- `POST /vouchers/{code}/redeem` – entwertet einen aktiven Gutschein.
- `DELETE /vouchers/{code}` – löscht den Gutschein (einfaches Admin-Helferlein).
- `POST /claim` – Endpunkt für Endnutzer: gibt pro Endgerät (Cookie `voucher_device_id`) einen Gutschein zurück bzw. erstellt ihn, falls noch keiner existiert.

QR-Inhalt ist `voucher:<code>` und kann direkt im Redeem-Frontend gescannt werden.

## Daten & Ablage

- SQLite DB liegt unter `data/vouchers.db`.
- Stub-E-Mails werden als Textdatei unter `outbox/` abgelegt (inkl. Base64 der QR-Grafik).
- Statische Dateien liegen in `static/` und werden von FastAPI ausgeliefert.

## Hinweise

- CORS ist offen (`*`), damit Mobile-Clients direkt die API nutzen können.
- Die QR-Grafik kann auf dem Handy gespeichert oder aus der API geladen werden.
- Keine echte E-Mail-Übermittlung; Versand ist absichtlich ein lokaler Stub für schnelle Tests.***
