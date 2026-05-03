# Стратегия переноса dav2d → rav2d

## Общий подход

Повторяем проверенную стратегию rav1d (Rust-порт dav1d):

1. **c2rust** для начальной транспиляции C → Rust (уродливый но рабочий код)
2. Постепенная **идиоматизация** — модуль за модулем
3. Asm остаётся as-is через **FFI**
4. Conformance-тесты на **каждом шаге**

## Структура крейтов

```
rav2d/
├── crates/
│   ├── rav2d/          # Основная библиотека (публичный API)
│   ├── rav2d-sys/      # FFI-биндинги к dav2d C/asm
│   └── rav2d-cli/      # CLI-утилита
├── dav2d/              # Git submodule — оригинальный C-код
└── tests/              # Conformance-тесты
```

### rav2d-sys
- Автоматическая сборка dav2d через meson (build.rs)
- `extern "C"` биндинги через bindgen
- Expose: DSP function tables, CPU detection, asm routines

### rav2d
- Safe Rust API
- Постепенная замена C-модулей Rust-имплементациями
- C-совместимый API через `#[no_mangle] extern "C"` для drop-in замены

### rav2d-cli
- Аналог tools/dav2d.c
- IVF/OBU/AnnexB input
- Y4M/YUV/MD5 output

## Решение проблемы битовой глубины

В C:
```c
#define BITDEPTH 8
#include "recon_tmpl.c"
```

В Rust — generics:
```rust
trait Pixel: Copy + Into<i32> + From<u8> {
    const BITDEPTH: u8;
    const MAX: Self;
}

impl Pixel for u8 {
    const BITDEPTH: u8 = 8;
    const MAX: u8 = 255;
}

impl Pixel for u16 {
    const BITDEPTH: u8 = 16;
    const MAX: u16 = 0xFFFF;
}

fn recon<P: Pixel>(ctx: &mut FrameCtx<P>, ...) { ... }
```

## Порядок переноса модулей

| Фаза | Модули | Недели |
|------|--------|-------:|
| 1 | Типы/структуры + FFI к asm | 2–3 |
| 2 | getbits + msac (энтропия) | 2–3 |
| 3 | obu.c (парсинг) | 3–4 |
| 4 | tables + cdf | 2–3 |
| 5 | decode.c + _tmpl.c (ядро) | 6–8 |
| 6 | thread_task.c (многопоточность) | 3–4 |
| 7 | lib.c (публичный API) | 2–3 |
| 8 | CLI/тесты | 2–3 |

## Зависимости между модулями

```
FFI/Types → getbits → msac → obu → tables/cdf → decode → thread_task → lib → cli
```

Каждый модуль верифицируется conformance-тестами из dav2d-test-data.

## Многопоточность

Оригинал (C): кастомный thread pool + pthread + condition variables
- Фрейм-параллелизм
- Тайл-параллелизм
- Строчный параллелизм

Rust-версия:
- std::thread + Mutex/Condvar (минимальные изменения)
- Или crossbeam (scoped threads, каналы)
- Или rayon (для data-параллелизма)
- Borrow checker поможет ловить data races на этапе компиляции
