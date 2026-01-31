# PHD Stefan briquette

This repository contains a CUDA solver for a three-dimensional Stefan problem in a single briquette geometry.

The code models transient heat transfer with temperature-dependent material properties and an effective latent-heat term around the freezing point. The finite-difference update is executed on the GPU over a regular 3D grid.

The project is research code rather than a general simulation package: experiment parameters are defined directly in the source, and results are written as a text summary together with VTK snapshots for later visualization.

## Описание

Этот репозиторий содержит CUDA решатель трехмерной задачи Стефана для геометрии с одним брикетом.

Код моделирует нестационарный теплоперенос с зависящими от температуры свойствами материала и эффективным учетом скрытой теплоты вблизи точки замерзания. Конечно-разностное обновление выполняется на GPU по регулярной трехмерной сетке.

Это исследовательский код, а не универсальный пакет моделирования: параметры эксперимента задаются непосредственно в исходниках, а результаты сохраняются в виде текстовой сводки и снимков VTK для последующей визуализации.

## Сборка

Нужны CMake, CUDA Toolkit и совместимый NVIDIA GPU.

```sh
cmake --preset release
cmake --build --preset release
```

Исполняемый файл:

```text
build/release/briquette
```

## Запуск

Перед запуском создайте каталог для VTK-файлов:

```sh
mkdir -p plot
./build/release/briquette
```

Расчет не принимает аргументы командной строки. Размер брикета, температура окружающей среды, шаги по пространству и времени и остальные параметры задаются в `src/briquette.cu`.

## Результат

`out.txt` содержит временную зависимость температуры в центре брикета. Снимки трехмерного температурного поля сохраняются в:

```text
plot/result_<time>.vtk
```

Их можно открыть в ParaView или другом VTK-совместимом инструменте.

## Модель

В коде используется трехмерная явная конечно-разностная схема. Теплоемкость и теплопроводность зависят от температуры, а скрытая теплота фазового перехода учитывается через изменение эффективной теплоемкости около температуры замерзания.
