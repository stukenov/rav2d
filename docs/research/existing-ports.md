# Существующие порты и биндинги dav1d

Дата: 2026-05-03

## Полные переписывания dav1d

| Проект | Язык | Звёзды | Тип | Статус |
|--------|------|-------:|-----|--------|
| [memorysafety/rav1d](https://github.com/memorysafety/rav1d) | Rust + asm | 623 | Полный порт dav1d | Зрелый, активный, ISRG |
| [imazen/rav1d-safe](https://github.com/imazen/rav1d-safe) | Rust | 3 | Форк: 160K строк asm → safe Rust intrinsics | Экспериментальный |
| [rainliu/rav1d](https://github.com/rainliu/rav1d) | Rust | 15 | Независимая попытка | Заглох (2025) |

## FFI-обёртки dav1d

### Rust
| Проект | Звёзды | Примечание |
|--------|-------:|------------|
| [rust-av/dav1d-rs](https://github.com/rust-av/dav1d-rs) | 58 | Основные биндинги. dav1d-sys: 959K downloads |
| [shiguredo/dav1d-rs](https://github.com/shiguredo/dav1d-rs) | 1 | Альтернативные биндинги |

### Swift / Apple
| Проект | Звёзды | Примечание |
|--------|-------:|------------|
| [SDWebImage/libdav1d-Xcode](https://github.com/SDWebImage/libdav1d-Xcode) | 14 | CocoaPods, SPM, Carthage |
| [awxkee/avif.swift](https://github.com/awxkee/avif.swift) | 60 | AVIF через dav1d для iOS/macOS |

### JavaScript / WASM
| Проект | Звёзды | Примечание |
|--------|-------:|------------|
| [Kagami/dav1d.js](https://github.com/Kagami/dav1d.js) | 35 | dav1d → WASM через Emscripten |
| [Kagami/avif.js](https://github.com/Kagami/avif.js) | 706 | AVIF полифилл через dav1d |
| [bvibber/ogv.js](https://github.com/bvibber/ogv.js) | 1,238 | Медиаплеер с dav1d для AV1 |

### Android / Java
| Проект | Звёзды | Примечание |
|--------|-------:|------------|
| [androidx/media decoder_av1](https://github.com/androidx/media) | — | Официальный Google Media3. Заменил libgav1 на dav1d |
| [awxkee/avif-coder](https://github.com/awxkee/avif-coder) | 95 | AVIF для Android |

### Другие
| Проект | Язык | Примечание |
|--------|------|------------|
| [MoonsideGames/dav1dfile](https://github.com/MoonsideGames/dav1dfile) | C | Обёртка для геймдевов |
| ferus-web/dav1d | Nim | WIP биндинги |
| capocasa/nim-dav1d | Nim | Обёртка |

## Независимые AV1 декодеры (не порты dav1d)

| Проект | Язык | Примечание |
|--------|------|------------|
| libgav1 (Google) | C++17 | Значительно медленнее dav1d |
| libaom (AOM) | C | Референсная имплементация |
| SVT-AV1 (Intel/Netflix) | C | Фокус на threading |

## AV2 (dav2d) — существующие проекты

| Проект | Тип | Статус |
|--------|-----|--------|
| [rust-av/dav2d-rs](https://github.com/rust-av/dav2d-rs) | FFI-обёртка dav2d | Создан 2026-05-03, 3 коммита |
| [rafaelcaricio/av2_demo](https://github.com/rafaelcaricio/av2_demo) | AV2 КОДИРОВЩИК (WASM) | Демо, март 2026 |
| **rav2d** | — | **Не существует. Имя свободно.** |

## Пробелы — биндинги НЕ существуют для:

Go, Zig, C#/.NET, Dart/Flutter, Python (standalone), Kotlin Multiplatform, PHP, Ruby, Haskell

## Ключевой референс: rav1d

- Подход: c2rust транспиляция → постепенная идиоматизация
- Asm: оставлен без изменений через FFI
- Производительность: ~5% медленнее C
- Баунти: $20,000 за оптимизацию
- Подрядчик: Immunant
- Лицензия: BSD-2-Clause
