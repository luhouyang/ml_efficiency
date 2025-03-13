# ML Efficiency

Trying out model training speed ups and memory efficiency techniques

## cProfile & pstats

[example](/tests/cp_metrics.py)

## vizualize with snakeviz

```console
pip install snakeviz
```

1. Configure Jupyter Notebook

    ```console
    conda install ipykernel
    python -m ipykernel install --user --name=YOUR_ENV_NAME
    ```

2. Start

    ```console
    jupyter notebook
    ```

3. Change `kernel` to your kernel

## timeit

[example](/tests/timeit_metrics.py)

## memory_profiler

```console
pip install https://github.com/fabianp/memory_profiler/archive/fix_195.zip
```

## line_profile

```console
pip install line_profiler[ipython]
```