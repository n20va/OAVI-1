from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

M_DEFAULT = 3
N_DEFAULT = 2

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR_NAME = "results"
SRC_DIR_NAME = "src"
REPORT_NAME = "report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Лабораторная работа №1: цветовые модели и передискретизация изображений"
    )
    parser.add_argument(
        "input_image",
        nargs="?",
        default=None,
        help="Путь к исходному изображению (png или bmp). Если не указан, скрипт попробует найти image.png рядом с собой.",
    )
    parser.add_argument("--m", type=int, default=M_DEFAULT, help="Коэффициент растяжения M")
    parser.add_argument("--n", type=int, default=N_DEFAULT, help="Коэффициент сжатия N")
    parser.add_argument(
        "--output-dir",
        default=str(BASE_DIR),
        help="Папка, в которой будут созданы results, src и report.md",
    )
    return parser.parse_args()


def resolve_input_image(path_str: str | None, base_dir: Path) -> Path:
    if path_str:
        path = Path(path_str).expanduser()
        return path if path.is_absolute() else (Path.cwd() / path).resolve()

    candidates = [
        base_dir / "image.png",
        base_dir / "source.png",
        base_dir / "input.png",
        base_dir / "image.bmp",
        base_dir / "source.bmp",
        base_dir / "input.bmp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Не найдено исходное изображение. Поместите рядом со скриптом файл image.png "
        "или запустите программу так: python lab1.py путь_к_изображению"
    )


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_rgb(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB").save(path)


def save_gray(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="L").save(path)


def rgb_channel_images(source: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_comp = np.zeros_like(source)
    g_comp = np.zeros_like(source)
    b_comp = np.zeros_like(source)
    r_comp[..., 0] = source[..., 0]
    g_comp[..., 1] = source[..., 1]
    b_comp[..., 2] = source[..., 2]
    return r_comp, g_comp, b_comp


def rgb_to_hsi(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_norm = rgb.astype(np.float64) / 255.0
    r = rgb_norm[..., 0]
    g = rgb_norm[..., 1]
    b = rgb_norm[..., 2]

    intensity = (r + g + b) / 3.0

    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = np.zeros_like(intensity)
    mask = intensity > 1e-12
    saturation[mask] = 1.0 - min_rgb[mask] / intensity[mask]
    saturation = np.clip(saturation, 0.0, 1.0)

    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    denominator = np.where(denominator < 1e-12, 1e-12, denominator)
    theta = np.arccos(np.clip(numerator / denominator, -1.0, 1.0))
    hue = np.where(b <= g, theta, 2.0 * np.pi - theta)
    hue = np.mod(hue, 2.0 * np.pi)

    return hue, saturation, intensity


def hsi_to_rgb(hue: np.ndarray, saturation: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    h = np.mod(hue, 2.0 * np.pi)
    s = np.clip(saturation, 0.0, 1.0)
    i = np.clip(intensity, 0.0, 1.0)

    r = np.zeros_like(i)
    g = np.zeros_like(i)
    b = np.zeros_like(i)

    eps = 1e-12
    sector1 = (h >= 0) & (h < 2.0 * np.pi / 3.0)
    sector2 = (h >= 2.0 * np.pi / 3.0) & (h < 4.0 * np.pi / 3.0)
    sector3 = ~(sector1 | sector2)

    h1 = h[sector1]
    i1 = i[sector1]
    s1 = s[sector1]
    b[sector1] = i1 * (1.0 - s1)
    den1 = np.cos(np.pi / 3.0 - h1)
    den1 = np.where(np.abs(den1) < eps, eps, den1)
    r[sector1] = i1 * (1.0 + s1 * np.cos(h1) / den1)
    g[sector1] = 3.0 * i1 - (r[sector1] + b[sector1])

    h2 = h[sector2] - 2.0 * np.pi / 3.0
    i2 = i[sector2]
    s2 = s[sector2]
    r[sector2] = i2 * (1.0 - s2)
    den2 = np.cos(np.pi / 3.0 - h2)
    den2 = np.where(np.abs(den2) < eps, eps, den2)
    g[sector2] = i2 * (1.0 + s2 * np.cos(h2) / den2)
    b[sector2] = 3.0 * i2 - (r[sector2] + g[sector2])

    h3 = h[sector3] - 4.0 * np.pi / 3.0
    i3 = i[sector3]
    s3 = s[sector3]
    g[sector3] = i3 * (1.0 - s3)
    den3 = np.cos(np.pi / 3.0 - h3)
    den3 = np.where(np.abs(den3) < eps, eps, den3)
    b[sector3] = i3 * (1.0 + s3 * np.cos(h3) / den3)
    r[sector3] = 3.0 * i3 - (g[sector3] + b[sector3])

    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def bilinear_resize(image: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w, _ = image.shape
    if new_h <= 0 or new_w <= 0:
        raise ValueError("Размер выходного изображения должен быть положительным.")
    if src_h == new_h and src_w == new_w:
        return image.copy()

    image_f = image.astype(np.float64)
    y = np.linspace(0, src_h - 1, new_h) if new_h > 1 else np.array([0.0])
    x = np.linspace(0, src_w - 1, new_w) if new_w > 1 else np.array([0.0])

    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)

    wy = y - y0
    wx = x - x0

    wa = (1.0 - wy)[:, None] * (1.0 - wx)[None, :]
    wb = wy[:, None] * (1.0 - wx)[None, :]
    wc = (1.0 - wy)[:, None] * wx[None, :]
    wd = wy[:, None] * wx[None, :]

    top_left = image_f[y0[:, None], x0[None, :]]
    bottom_left = image_f[y1[:, None], x0[None, :]]
    top_right = image_f[y0[:, None], x1[None, :]]
    bottom_right = image_f[y1[:, None], x1[None, :]]

    result = (
        top_left * wa[..., None]
        + bottom_left * wb[..., None]
        + top_right * wc[..., None]
        + bottom_right * wd[..., None]
    )
    return np.clip(result, 0, 255).round().astype(np.uint8)


def stretch_interpolation(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 0:
        raise ValueError("Коэффициент растяжения должен быть положительным.")
    src_h, src_w, _ = image.shape
    return bilinear_resize(image, max(1, round(src_h * factor)), max(1, round(src_w * factor)))


def decimation(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 0:
        raise ValueError("Коэффициент сжатия должен быть положительным.")
    return image[::factor, ::factor].copy()


def one_pass_resample(image: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 0:
        raise ValueError("Коэффициент передискретизации должен быть положительным.")
    src_h, src_w, _ = image.shape
    return bilinear_resize(image, max(1, round(src_h * factor)), max(1, round(src_w * factor)))


def format_size(shape: tuple[int, int, int]) -> str:
    return f"{shape[1]}x{shape[0]}"


def write_report(
    report_path: Path,
    input_image: Path,
    m: int,
    n: int,
    source_shape: tuple[int, int, int],
    stretched_shape: tuple[int, int, int],
    decimated_shape: tuple[int, int, int],
    two_pass_shape: tuple[int, int, int],
    one_pass_shape: tuple[int, int, int],
) -> None:
    k = m / n
    report = f"""# Лабораторная работа №1
## Цветовые модели и передискретизация изображений

### Цель работы
Изучить представление изображения в цветовой модели RGB и HSI, а также реализовать основные операции передискретизации без использования библиотечных функций масштабирования.

### Исходные данные
- Исходный файл: `{input_image.name}`
- Коэффициент растяжения: `M = {m}`
- Коэффициент сжатия: `N = {n}`
- Итоговый коэффициент передискретизации: `K = M/N = {k:.3f}`

### Исходное изображение
![Исходное изображение](src/source.png)

## 1. Цветовые модели

### 1.1 Компоненты R, G, B
| Красный канал | Зеленый канал | Синий канал |
|:-------------:|:-------------:|:-----------:|
| ![R](src/r_channel.png) | ![G](src/g_channel.png) | ![B](src/b_channel.png) |

### 1.2 Яркостная компонента HSI
Яркостная компонента `I` получена после преобразования исходного изображения из RGB в HSI.

![Яркостная компонента HSI](src/intensity_channel.png)

### 1.3 Инвертирование яркостной компоненты
В данном пункте была выполнена инверсия яркости по формуле `I' = 1 - I`, после чего изображение было преобразовано обратно в RGB.

| Исходное изображение | Изображение после инверсии яркости |
|:--------------------:|:----------------------------------:|
| ![Исходное](src/source.png) | ![Инвертированное](src/inverted_intensity.png) |

## 2. Передискретизация

### 2.1 Растяжение в M раз
Для увеличения размера изображения использована билинейная интерполяция.

| Исходное изображение | Растянутое изображение |
|:--------------------:|:----------------------:|
| ![Исходное](src/source.png) | ![Растянутое](src/upscaled.png) |

### 2.2 Сжатие в N раз
Для уменьшения изображения использован метод прореживания: выбирался каждый `N`-й пиксель по строкам и столбцам.

| Исходное изображение | Сжатое изображение |
|:--------------------:|:------------------:|
| ![Исходное](src/source.png) | ![Сжатое](src/downscaled.png) |

### 2.3 Двухпроходная передискретизация в K = M/N
На первом шаге выполнялось растяжение в `M` раз, на втором — сжатие результата в `N` раз.

| Исходное изображение | Результат двухпроходной обработки |
|:--------------------:|:---------------------------------:|
| ![Исходное](src/source.png) | ![Два прохода](src/two_pass.png) |

### 2.4 Однопроходная передискретизация в K
Изменение размера выполнено за один проход непосредственно с коэффициентом `K = {k:.3f}`.

| Исходное изображение | Результат однопроходной обработки |
|:--------------------:|:---------------------------------:|
| ![Исходное](src/source.png) | ![Один проход](src/one_pass.png) |

## Результаты выполнения

| Операция | Размер изображения |
|:---------|-------------------:|
| Исходное изображение | {format_size(source_shape)} |
| Растяжение в `M={m}` | {format_size(stretched_shape)} |
| Сжатие в `N={n}` | {format_size(decimated_shape)} |
| Двухпроходная передискретизация | {format_size(two_pass_shape)} |
| Однопроходная передискретизация | {format_size(one_pass_shape)} |

## Вывод
В ходе выполнения лабораторной работы были реализованы операции выделения цветовых каналов в модели RGB, вычисления яркостной компоненты в модели HSI и инвертирования яркости. Также были реализованы методы растяжения, сжатия и два варианта передискретизации изображения: двухпроходный и однопроходный. Полученные результаты подтверждают корректность выполненных преобразований и различие подходов к изменению размера изображения.
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.m <= 0 or args.n <= 0:
        raise ValueError("Параметры M и N должны быть положительными целыми числами.")

    base_output_dir = Path(args.output_dir).expanduser()
    if not base_output_dir.is_absolute():
        base_output_dir = (Path.cwd() / base_output_dir).resolve()

    results_dir = base_output_dir / RESULTS_DIR_NAME
    src_dir = base_output_dir / SRC_DIR_NAME
    report_path = base_output_dir / REPORT_NAME
    results_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    input_image = resolve_input_image(args.input_image, BASE_DIR)
    source = load_rgb(input_image)

    save_rgb(source, results_dir / "source.png")
    save_rgb(source, src_dir / "source.png")

    r_comp, g_comp, b_comp = rgb_channel_images(source)
    save_rgb(r_comp, results_dir / "component_r.png")
    save_rgb(g_comp, results_dir / "component_g.png")
    save_rgb(b_comp, results_dir / "component_b.png")
    save_rgb(r_comp, src_dir / "r_channel.png")
    save_rgb(g_comp, src_dir / "g_channel.png")
    save_rgb(b_comp, src_dir / "b_channel.png")

    h, s, intensity = rgb_to_hsi(source)
    intensity_img = np.clip(intensity * 255.0, 0, 255).round().astype(np.uint8)
    save_gray(intensity_img, results_dir / "intensity_hsi.png")
    save_gray(intensity_img, src_dir / "intensity_channel.png")

    inverted_rgb = hsi_to_rgb(h, s, 1.0 - intensity)
    save_rgb(inverted_rgb, results_dir / "intensity_inverted_rgb.png")
    save_rgb(inverted_rgb, src_dir / "inverted_intensity.png")

    stretched = stretch_interpolation(source, args.m)
    save_rgb(stretched, results_dir / "stretch_m.png")
    save_rgb(stretched, src_dir / "upscaled.png")

    decimated = decimation(source, args.n)
    save_rgb(decimated, results_dir / "decimation_n.png")
    save_rgb(decimated, src_dir / "downscaled.png")

    two_pass = decimation(stretched, args.n)
    save_rgb(two_pass, results_dir / "resample_two_pass.png")
    save_rgb(two_pass, src_dir / "two_pass.png")

    k = args.m / args.n
    one_pass = one_pass_resample(source, k)
    save_rgb(one_pass, results_dir / "resample_one_pass.png")
    save_rgb(one_pass, src_dir / "one_pass.png")

    write_report(
        report_path=report_path,
        input_image=input_image,
        m=args.m,
        n=args.n,
        source_shape=source.shape,
        stretched_shape=stretched.shape,
        decimated_shape=decimated.shape,
        two_pass_shape=two_pass.shape,
        one_pass_shape=one_pass.shape,
    )

    print("Лабораторная работа выполнена успешно.")
    print(f"Исходное изображение: {input_image}")
    print(f"Папка с результатами: {results_dir}")
    print(f"Папка с изображениями для отчёта: {src_dir}")
    print(f"Отчёт: {report_path}")


if __name__ == "__main__":
    main()
