# uztts_text

O'zbekcha text frontend — modeldan butunlay mustaqil (CLAUDE.md §5).
Kiruvchi: xom matn (lotin yoki kirill). Chiquvchi: normallashgan matn
(lotin, kichik harf, okina ʻ / tutuq ʼ, sonlar so'z bilan, punktuatsiyasiz).

```bash
uv run uztts-text normalize "Мен 25 ёшдаман"
# men yigirma besh yoshdaman
```

## Modullar

| Modul | Qamrov |
|---|---|
| `apostrophes.py` | barcha apostrof variantlari → okina (oʻ/gʻ) yoki tutuq (ʼ) |
| `translit.py` | kirill → lotin, kontekstli е (ye/e), ъ→ʼ, ў/қ/ғ/ҳ |
| `numbers.py` | butun sonlar (kvadrilliongacha), ming ajratkichlari (bo'shliq/vergul), % → foiz |
| `normalize.py` | to'liq quvur: apostrof → translit → sonlar → kichik harf → punktuatsiya |

## Qoidalar

- **Har bir qoida uchun test majburiy**: `tests/text/golden.jsonl` —
  kirish/chiqish juftliklari, bu fayl faqat o'sadi.
- Son o'qilishi tabiiy shaklda: `100` → "yuz" (bir yuz emas), `1000` →
  "ming", `1995` → "ming toʻqqiz yuz toʻqson besh", `1000000` → "bir million".
- `1 245 000` / `1,245,000` — ming ajratkichi sifatida o'qiladi.

## Hali yo'q (keyingi bosqichlar)

Kasr sonlar, valyuta, sana/vaqt, telefon raqamlari, qisqartmalar (INN,
PINFL), lotin→kirill teskari yo'nalish, klitika kanonizatsiyasi
(`qildik-da` konventsiyasi — ASR o'lchovi uchun, alohida qaror bilan).
