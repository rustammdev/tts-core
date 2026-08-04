# uz-tts

O'zbek tili uchun hissiy jihatdan ifodali TTS. Hozirgi holat — V0 skelet:
data kontrakti va manifest asboblari.

Loyiha qarorlari va yo'l xaritasi: [CLAUDE.md](CLAUDE.md).

## Boshlash

```bash
make setup     # uv sync
make check     # lint + typecheck + test
```

## Manifest bilan ishlash

```bash
uztts-data validate data/manifests/train.jsonl
uztts-data hash     data/manifests/train.jsonl
```

`validate` sxema xatolarini va takrorlangan `id` larni satr raqami bilan
ko'rsatadi, xato bo'lsa 1 kod bilan chiqadi. `hash` — train run'ini qaysi data
bilan bog'lash uchun sha256.

## Tuzilishi

| Yo'l | Nima |
|---|---|
| `src/uztts_data/` | data kontrakti, manifest I/O, CLI |
| `data/` | git'ga kirmaydi; `raw` → `interim` → `processed`, manifestlar `manifests/` da |
| `docs/decisions/` | qabul qilingan qarorlar |
