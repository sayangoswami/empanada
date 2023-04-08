# empanada
Efficient massively-parallel nvidia-accelerated de novo assembler

## Build

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
```

## Run

```bash
./empanada input.fastq output.bin \
  -o 75  \ # (minimum overlap length, default = 31) \
  -m 8g  \ # (device block size during fingerprint generation, default = 1g, tested with 8g on a 32GB card)
  -r 12g \ # (device block size during overlap detection, default = 1g, tested with 12g on a 32GB card)
  -b 16m \ # (block size for transposed bases, default = 32m)
  -c 64m   #(block size for key-value pairs, default = 64m)
```

## Attributions:

* https://github.com/attractivechaos/klib
