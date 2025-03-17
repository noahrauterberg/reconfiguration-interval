# The Impact of Starlink's Reconfiguration Interval on Network Routing

Placement algorithms, simulation environment, and analysis tools for analyzing Starlink's reconfiguration interval.

If you use this software in a publication, please cite it as:

## Text

Noah Rauterberg, **Analyzing the Impact of Starlink’s Reconfiguration Intervals on Network Routing**, Bachelor's Thesis, Berlin, Germany, 2025

```bibtex
@thesis{rauterberg_reconfiguratio_interval:_2025,
    title = "Analyzing the Impact of Starlink’s Reconfiguration Intervals on Network Routing",
    author = "Rauterberg, Noah",
    year = 2025,
}
```

## License

The code in this repository is licensed under the terms of the [MIT](./LICENSE) license.

All code in the `simulation` folder is based on the [SILLEO-SCNS project](https://github.com/Ben-Kempton/SILLEO-SCNS) and licensed under the [GNU General Public License Version 3](./simulation/LICENSE).
This `README` file as well as the `config.py` and `distances.py` files are largely based on the [QoS-Aware Resource Placement for the LEO Edge](https://github.com/pfandzelter/optimal-leo-placement) and licensed under the [MIT LICENSE](./LICENSE).

## Usage

This project requires Python 3.9 or later.
Install the required packages with `pip install -r requirements.txt`, a virtual environment is recommended.
Alternatively, use the provided Dockerfile and run all subsequent commands inside the container:

```sh
docker build -t py .
docker run --rm -it -v "$(pwd)":/run py
cd /run
```

Please bear in mind that the code in this repository is not optimized for performance.
Memory, disk, and CPU usage can be high, especially for larger constellations and with increased simulation granularity.
Further, determining GSLs for many ground stations at once takes a lot of time, especially if there cannot be a single GSL for an interval.

### Configuration

Configure simulation parameters in `config.py`.
This includes parameters for the shells as well as service level objectives (SLO).

### Constellation Simulation

The constellation simulation yields ISL distances, positions as well as GS positions for the configured shells.
Note that GS positions are calculated for each shell, even though the same GSs are used.
Run `python3 distances.py` to generate these distances.

### Simulate the reconfiguration interval

To simulate paths from GSs to certain 'server'-GSs, run `python3 analysis.py`.
Note that by default, the server-GSs are placed at `(0, 0)`, `(0, 25)`, `(0, 50)` and must be configured by hand if other targets should be used.

### Analysis

To analyze the results, several functions are defined in `plot.py`, but must be manually called.
