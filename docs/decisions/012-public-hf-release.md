# 012 — HF repolarini ochiq qilish va professional nomlash

Sana: 2026-09-03 · Holat: qabul qilingan

## Kontekst

STT yo'nalishi (birinchi mahsulot) ko'rsatiladigan natijaga yetdi: olti
checkpoint, o'lchov to'plami, manifest kontrakti. Foydalanuvchi shu ishni
tashqi auditoriyaga (VoiceLab) ko'rsatmoqchi. Eski holat buni ko'tarmasdi:
repolar `uz-stt` / `uz-stt-data` / `uz-stt-etalon-v2` deb nomlangan, hammasi
private, kartalar o'zbekcha va "private" deb belgilangan, `uz-stt` kartasi
2026-08-10 dagi holatda qotib qolgan (600M yo'q).

## Qaror

**Nomlash** — `til-vazifa-arxitektura` tartibi (HF konventsiyasi):

| Eski | Yangi | Turi |
|---|---|---|
| `uz-stt` | `uzbek-asr-gigaam` | model |
| `uz-stt-data` | `uzbek-asr-train-manifests` | dataset |
| `uz-stt-etalon-v2` | `uzbek-asr-benchmark-spontaneous` | dataset |

`uz-stt-etalon` (v1, faqat 14 KB corrections) va `testmodel` o'chirildi.
HF eski nomdan yangisiga yo'naltiradi, shuning uchun eski havolalar buzilmaydi.

**Ochiqlik** — uchala repo public. Xavfsizlik chegarasi o'zgarmadi:

- Model og'irliklari — MIT (baza GigaAM-Multilingual'dan meros).
- Manifestlar — **ko'rsatkich**, audio emas; har satr yuqori oqim parquet
  fayli va satriga ishora qiladi.
- Etalon — faqat matn va satr xaritasi. **Audio tarqatilmaydi**;
  `reconstruct_benchmark_audio.py` uni yuqori oqim datasetidan qayta yig'adi.

Ya'ni CLAUDE.md §2 dagi "xom YouTube audio hech qachon public qilinmaydi"
qoidasi kuchida qoladi.

**Kartalar inglizcha va repoda saqlanadi.** Manba — `hub/` papkasi,
`uztts-asr hub push-cards` bilan yuklanadi; HF veb-UI'da tahrir qilinmaydi.
Sabab: karta ham artefakt, versiyalanishi va testdan o'tishi kerak
(`tests/asr/test_hub.py` har e'lon qilinadigan fayl borligini va model
kartasi `inference.py` taklif qiladigan checkpointlarni sanashini tekshiradi).

**E'lon qilinadigan skriptlar mustaqil.** `inference.py`, `scoring.py`,
`reconstruct_benchmark_audio.py` `uztts_*` paketlarini import qilmaydi —
kod reposi private, foydalanuvchi faqat HF fayllarini ko'radi. Bu takror
(normalize_text, join_clitics, CTC bosh kengaytmasi ikki joyda) —
ongli narx; `hub/README.md` da qayd etilgan.

## Muqobillar

- **Har o'lcham uchun alohida repo** (`...-600m`, `...-220m`) — rad:
  ablatsiya narigi checkpointlarsiz o'qilmaydi, 8 GB qayta yuklash kerak
  bo'lardi, va nom faqat bitta o'lchamni aytib qolganini chalg'itardi.
- **Kartalarni koddan generatsiya qilish** (avvalgi `model_card()` funksiyasi)
  — rad: karta endi uzun matn, f-string ichida yozish o'qib bo'lmaydigan
  bo'lardi. Fayl + test yaxshiroq.
- **Etalon audiosini ham yuklash** — rad: YouTube audiosi, egasi boshqa;
  qayta yig'ish skripti yetarli.

## Oqibatlar

- `uztts_asr.hub` uchta repo bilan ishlaydi (`ensure_repos` endi 3 qiymat
  qaytaradi), `dataset_card()` / `model_card()` funksiyalari olib tashlandi.
- `push-data` endi README yozmaydi — faqat manifestlar va `stats.json`;
  karta `push-cards` bilan alohida boradi.
- Kod reposi (github.com/rustammdev/tts-core) hozircha private.
