# uztts_asr

O'z ASR modelimiz: GigaAM'ni o'zbekchaga fine-tune qilish (007-qaror).
Transkripsiya — TTS data sifatining shifti; bu paket o'sha shiftni ko'taradi.

## Modullar

| Modul | Nima |
|---|---|
| `prepare.py` | 7 korpusni yagona trening manifestiga keltirish, `uztts-asr prepare` |

## prepare

```bash
uztts-asr prepare                 # hamma manbalar
uztts-asr prepare --only fleurs --only common_voice
uztts-asr prepare --limit 100     # sinov
```

Kirish: `$UZTTS_DATA_ROOT/train_corpora/` dagi 7 korpus (USC, Common
Voice, UzbekVoice, 3×YouTube, FLEURS). Chiqish:
`$UZTTS_DATA_ROOT/asr/{train,val,test}_manifest.jsonl`.

**Diskni tejash:** audio nusxalanmaydi — manifest satri parquet fayl +
qator raqamiga ko'rsatadi (`{"parquet": ..., "row": N}`), trening
dataloader'i audioni o'qish paytida dekodlaydi. Istisno — FLEURS: tar
ichidan chiqariladi (`audio_filepath`, ~1.5 GB).

Har satr: `duration`, `text` (normallashgan: kichik harf, okina ʻ,
punktuatsiyasiz), `text_raw` (asl, punktuatsiyali — punktuatsiya fazasi
uchun), `source`, `split`.

Filtrlar (uzinfocom 701h retsepti): raqamli namunalar chiqariladi,
yot alifbo chiqariladi, 0.5–30 s, ≤18 belgi/s, 0.3–2.67 so'z/s.
Split spiker bo'yicha deterministik hash (96/2/2); FLEURS'ning dev/test
bo'linmasi benchmark sifatida o'zgarishsiz saqlanadi.

Idempotent: har parquet uchun tayyor shard qayta ishlanmaydi; yakunda
shardlar uch manifestga birlashtiriladi.
