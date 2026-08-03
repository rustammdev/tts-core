# 002 — Repo skeleti: faqat ishlaydigan paketlar

## Kontekst

Dastlabki reja `src/` ostida beshta paketni (`uztts_text`, `uztts_data`,
`uztts_train`, `uztts_eval`, `uztts_serve`) darhol yaratishni ko'zda tutgan edi.
V0 da ularning to'rttasida kod yo'q.

## Qaror

Paket **faqat ishlaydigan kod paydo bo'lganda** yaratiladi. Hozir `uztts_data`
bor. Rejadagi paketlar CLAUDE.md §3 jadvalida yozib qo'yilgan.

Qo'shimcha tanlovlar:

- **Bitta distributsiya (`uz-tts`), bir nechta top-level paket.** `uztts_text`
  o'z import ildizida qoladi — modeldan mustaqilligi shu bilan qulflanadi va
  keyin alohida paketga ajratish arzon bo'ladi.
- **`uv` + `hatchling`**, `src/` layout, Python 3.11 (`.python-version`).
- **`typer`** (MIT) CLI uchun — har bir pipeline bosqichi alohida buyruq
  bo'ladi, type hint'lardan interfeys o'zi hosil bo'ladi.
- **`configs/` hozir yo'q** — birinchi YAML konfig train bilan birga keladi.

## Muqobillar

| Variant | Nega yo'q |
|---|---|
| Bo'sh stub paketlar + README | O'qiladi, lekin ishlamaydi; eskiradi va yolg'on tuzilish beradi |
| Yagona `uztts` paketi, ichida submodullar | `uztts_text` mustaqilligini kod darajasida ushlab turmaydi |
| Har paket alohida distributsiya (workspace) | V0 uchun ortiqcha; bitta venv yetadi |
| `argparse` | Har bir bosqich CLI'sida qo'lda parsing — type hint'lar takrorlanadi |

## Oqibatlar

- CLAUDE.md §3 dagi jadval yagona reja manbai; yangi paket qo'shilganda holati
  yangilanadi.
- `make check` = lint + `mypy --strict` + `pytest`. Yangi paket shu uchtasidan
  o'tmasa qo'shilmaydi.
- Paket ajratish kerak bo'lsa (`uztts_text` ni alohida chiqarish) —
  `pyproject.toml` da bitta qator o'zgaradi.
