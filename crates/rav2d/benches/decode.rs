use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rav2d::{Decoder, Settings, version};

fn bench_version(c: &mut Criterion) {
    c.bench_function("version", |b| {
        b.iter(|| {
            black_box(version());
        });
    });
}

fn bench_decoder_open(c: &mut Criterion) {
    c.bench_function("Decoder::open default settings", |b| {
        let settings = Settings::default();
        b.iter(|| {
            let decoder = Decoder::open(black_box(&settings));
            black_box(decoder).ok();
        });
    });
}

fn bench_send_data_get_picture(c: &mut Criterion) {
    c.bench_function("Decoder::send_data + get_picture stub round-trip", |b| {
        b.iter(|| {
            let settings = Settings::default();
            let mut decoder = Decoder::open(&settings).expect("open failed");
            let _ = black_box(decoder.send_data(None));
            let _ = black_box(decoder.get_picture());
        });
    });
}

criterion_group!(
    benches,
    bench_version,
    bench_decoder_open,
    bench_send_data_get_picture
);
criterion_main!(benches);
