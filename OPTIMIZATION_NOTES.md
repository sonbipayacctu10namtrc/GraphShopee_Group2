# Optimization Notes

File nay dung de ghi lai moi lan toi uu solver:
- ngay thuc hien
- file/hang muc da thay doi
- ly do thay doi
- lenh benchmark da chay
- ket qua truoc/sau

## 2026-05-26 - MAPD-CBS pickup insertion heuristic

### Thay doi
- File: `solvers/mapd_cbs_solver.py`
- Them logic `insertion_penalty` khi shipper dang mang hang ma can quyet dinh co nen ghe pickup don moi hay khong.
- Them `_carried_orders()` va `_carried_delivery_cost()` de uoc luong chi phi chen pickup vao tuyen hien tai.
- Dieu chinh `_pickup_key()` de uu tien pickup khong lam lech qua nhieu tuyen giao hang dang mang.
- Dieu chinh `_should_pickup_before_delivery()` de can bang giua:
  - giao don dang mang dung han
  - ghe pickup don moi neu detour nho
  - uu tien don co priority cao hon
- Dieu chinh thu tu xu ly shipper tren map trung binh: shipper co don dang mang gap hon duoc quyet dinh truoc.

### Ly do
Truoc do solver de bi "ham pickup gan": shipper co the ghe lay them don moi nhung lam tre don dang mang. Thay doi nay giup solver danh gia chi phi chen pickup vao tuyen hien tai truoc khi quyet dinh.

### Benchmark
Lenh chay:

```powershell
$env:PYTHONIOENCODING='utf-8'; python run_test.py --config test_config.txt --out results --method MAPDCBSSolver
```

### Ket qua

Tong diem `MAPD-CBS`:

```text
Truoc: 3354.2899
Sau:   3519.4547
Tang:  +165.1648
```

Chi tiet sau toi uu:

| Config | Net reward | Delivered | On time | Late | Missed |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 247.6663 | 13/15 | 11 | 2 | 2 |
| C2 | 425.9418 | 23/25 | 20 | 3 | 2 |
| C3 | 453.6739 | 17/40 | 16 | 1 | 23 |
| C4 | 705.9902 | 39/60 | 33 | 6 | 21 |
| C5 | 1037.9401 | 67/80 | 54 | 13 | 13 |
| C6 | 648.2424 | 45/100 | 30 | 15 | 55 |

### Ghi chu
- Day la cai tien heuristic, khong thay doi cong thuc cham diem trong `env.py`.
- Lan thu rule "dong" dua tren slack/backlog/priority nhieu hon da bi tut diem, nen khong giu.
- Huong tiep theo de tranh overfit: tao benchmark random nhieu config/seed hon roi toi uu theo diem trung binh.

## 2026-05-26 - MAPD-CBS delivery priority and route-aware pickup

### Thay doi
- File: `solvers/mapd_cbs_solver.py`
- Dieu chinh `_delivery_key()`:
  - voi map nho/trung binh (`N < 18`), uu tien giao don priority cao hon sau khi xet co tre han hay khong.
  - voi map lon (`N >= 18`), giu deadline-first de bao toan kha nang giao nhieu don tren C5/C6.
- Them bo nho pickup gan day cho map trung binh (`15 <= N < 18`):
  - ghi lai pickup source cua cac don moi vua reveal.
  - neu shipper ranh va khong co don phu hop, di ve vung pickup vua xuat hien gan day.
  - khong ap dung cho map lon vi thu nghiem lam C5/C6 tut manh.
- Dieu chinh `_pickup_key()` cho map rat lon (`N >= 20`):
  - them thanh phan `pickup_dist + 0.5 * delivery_dist`.
  - muc tieu la tranh chon don co diem pickup gan nhung diem giao qua xa.

### Ly do
Sau lan toi uu truoc, C6 van con thap vi solver thuong chon don dua tren pickup gan, nhung duong giao sau do dai. Voi map lon nhieu vat can, tong tuyen pickup + delivery quan trong hon pickup distance rieng le.

Voi map nho/trung binh, chi phi di chuyen thap hon, nen giao don priority cao som hon co the tang reward ma khong lam vo deadline qua nhieu.

### Benchmark
Lenh chay:

```powershell
$env:PYTHONIOENCODING='utf-8'; python run_test.py --config test_config.txt --out results --method MAPDCBSSolver
```

### Ket qua

Tong diem `MAPD-CBS`:

```text
Truoc lan nay: 3519.4547
Sau lan nay:   3988.7552
Tang:          +469.3005
```

So voi baseline ban dau truoc cac toi uu:

```text
Baseline dau: 3354.2899
Hien tai:     3988.7552
Tang tong:    +634.4653
```

Chi tiet sau toi uu:

| Config | Net reward | Delivered | On time | Late | Missed |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 247.6663 | 13/15 | 11 | 2 | 2 |
| C2 | 425.9418 | 23/25 | 20 | 3 | 2 |
| C3 | 487.1835 | 19/40 | 18 | 1 | 21 |
| C4 | 763.0302 | 36/60 | 32 | 4 | 24 |
| C5 | 1037.9442 | 67/80 | 54 | 13 | 13 |
| C6 | 1026.9892 | 63/100 | 54 | 9 | 37 |

### Bien the da thu nhung khong giu
- Doi `_avoid_conflicts()` sang uu tien shipper co deadline gap: lam C5/C6 tut, vi khong khop voi thu tu xu ly va cham theo id cua env.
- Noi dieu kien pickup-before-delivery tren map lon: C6 tut manh.
- Idle reposition cho tat ca map lon: C4 tang nhung C5/C6 tut manh.
- Pickup priority-first: shipper bo qua don gan de chay xa lay priority cao, tong diem tut.
- He so delivery distance trong pickup key:
  - `0.35`: C6 tut.
  - `0.65`: C6 tut.
  - `0.5`: tot nhat trong cac lan thu.

## 2026-05-26 - Anti-overfit validation and broader route-aware pickup

### Thay doi
- Them `make_validation_config.py` de sinh `validation_config.txt` co dinh bang seed.
- Them `validation_config.txt` gom 8 config V1-V8 voi N/C/G/T va obstacle khac bo C1-C6.
- Dieu chinh `_pickup_key()` trong `solvers/mapd_cbs_solver.py`:
  - map rat lon `N >= 19`: dung `pickup_dist + 0.5 * delivery_dist`.
  - map trung binh nho `13 <= N < 15`: dung `pickup_dist + 0.25 * delivery_dist`.

### Ly do
Lan truoc route-aware pickup chi ap dung cho `N >= 20`, giup C6 nhung chua kiem tra ngoai bo public. Khi them validation, V8 co `N=19` cho thay cung can tinh ca delivery distance. Mo rong nguong tu `N >= 20` sang `N >= 19` tang validation ro ret ma khong lam thay doi diem public.

Voi map `13 <= N < 15`, validation V7 cho thay pickup gan nhung delivery xa lam solver bo lo nhieu don. Them mot phan nho delivery distance giup V7 tang manh, trong khi public khong bi anh huong vi khong co config N=13/14.

### Benchmark
Lenh public:

```powershell
$env:PYTHONIOENCODING='utf-8'; python run_test.py --config test_config.txt --out results --method MAPDCBSSolver
```

Lenh validation:

```powershell
$env:PYTHONIOENCODING='utf-8'; python run_test.py --config validation_config.txt --out results_validation --method MAPDCBSSolver
```

### Ket qua public

Tong diem `MAPD-CBS` tren `test_config.txt`:

```text
Truoc lan nay: 3988.7552
Sau lan nay:   3988.7552
Thay doi:      +0.0000
```

Chi tiet public sau toi uu:

| Config | Net reward | Delivered | On time | Late | Missed |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 247.6663 | 13/15 | 11 | 2 | 2 |
| C2 | 425.9418 | 23/25 | 20 | 3 | 2 |
| C3 | 487.1835 | 19/40 | 18 | 1 | 21 |
| C4 | 763.0302 | 36/60 | 32 | 4 | 24 |
| C5 | 1037.9442 | 67/80 | 54 | 13 | 13 |
| C6 | 1026.9892 | 63/100 | 54 | 9 | 37 |

### Ket qua validation

Tong diem `MAPD-CBS` tren `validation_config.txt`:

```text
Truoc lan nay: 4533.43
Sau lan nay:   4921.32
Tang:          +387.89
```

Chi tiet validation sau toi uu:

| Config | Net reward | Delivered | On time | Late | Missed |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1 | 382.25 | 15/18 | 15 | 0 | 3 |
| V2 | 683.27 | 30/32 | 29 | 1 | 2 |
| V3 | 157.51 | 9/45 | 9 | 0 | 36 |
| V4 | 600.79 | 40/65 | 29 | 11 | 25 |
| V5 | 642.94 | 38/85 | 34 | 4 | 47 |
| V6 | 1229.54 | 70/110 | 56 | 14 | 40 |
| V7 | 444.57 | 29/55 | 24 | 5 | 26 |
| V8 | 780.45 | 50/95 | 46 | 4 | 45 |

### Bien the da thu nhung khong giu
- Them delivery distance cho `N == 18` voi he so `0.15`: validation V5 tang manh nhung public C5 tut qua nhieu, nen bo.
- He so `0.15` cho `13 <= N < 15`: khong giu duoc muc tang cua V7, nen dung `0.25`.

### Ghi chu chong overfit
- Tu lan nay, moi toi uu nen chay ca `test_config.txt` va `validation_config.txt`.
- Chi nen giu thay doi khi public khong tut dang ke va validation khong xau di, hoac khi co ly do ro rang chap nhan trade-off.

## 2026-05-26 - Less overfit idle reposition and on-route pickup

### Thay doi
- File: `solvers/mapd_cbs_solver.py`
- Dieu chinh route-aware pickup cho map `13 <= N < 15`:
  - chi ap dung `pickup_dist + 0.25 * delivery_dist` khi `C >= 4`.
  - ly do: validation V3 (`N=13, C=3`) bi hai nhe khi ap dung route-aware pickup, trong khi V7 (`N=14, C=4`) duoc loi ro.
- Mo rong idle reposition:
  - tu `15 <= N < 18` thanh `13 <= N < 18` khi `C >= 3`.
  - shipper ranh di ve vung pickup moi quan sat gan day, nhung khong ap dung cho map nho `N=11` vi validation cho thay bi hai.
- Them rule pickup gan nhu nam tren duong giao cho map lon:
  - neu `N >= 19`, `via_pickup <= 1`, va `detour_extra <= 1`, cho phep ghe pickup ngay ca khi priority pickup thap hon don dang giao.
  - ly do: pickup gan nhu cung tuyen co chi phi co hoi rat nho, giup tang so don giao/dung han tren map lon.

### Benchmark
Lenh public:

```powershell
$env:PYTHONIOENCODING='utf-8'; python run_test.py --config test_config.txt --out results --method MAPDCBSSolver
```

Lenh validation:

```powershell
$env:PYTHONIOENCODING='utf-8'; python run_test.py --config validation_config.txt --out results_validation --method MAPDCBSSolver
```

### Ket qua public

Tong diem `MAPD-CBS` tren `test_config.txt`:

```text
Truoc lan nay: 3988.7552
Sau lan nay:   4144.1229
Tang:          +155.3677
```

Chi tiet public sau toi uu:

| Config | Net reward | Delivered | On time | Late | Missed |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 247.6663 | 13/15 | 11 | 2 | 2 |
| C2 | 425.9418 | 23/25 | 20 | 3 | 2 |
| C3 | 487.1835 | 19/40 | 18 | 1 | 21 |
| C4 | 763.0302 | 36/60 | 32 | 4 | 24 |
| C5 | 1037.9442 | 67/80 | 54 | 13 | 13 |
| C6 | 1182.3569 | 65/100 | 58 | 7 | 35 |

### Ket qua validation

Tong diem `MAPD-CBS` tren `validation_config.txt`:

```text
Truoc lan nay: 4921.32
Sau lan nay:   5613.8469
Tang:          +692.5269
```

Chi tiet validation sau toi uu:

| Config | Net reward | Delivered | On time | Late | Missed |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1 | 382.2529 | 15/18 | 15 | 0 | 3 |
| V2 | 683.2727 | 30/32 | 29 | 1 | 2 |
| V3 | 701.0398 | 33/45 | 30 | 3 | 12 |
| V4 | 600.7921 | 40/65 | 29 | 11 | 25 |
| V5 | 642.9352 | 38/85 | 34 | 4 | 47 |
| V6 | 1344.8663 | 72/110 | 64 | 8 | 38 |
| V7 | 478.2386 | 30/55 | 27 | 3 | 25 |
| V8 | 780.4493 | 50/95 | 46 | 4 | 45 |

### Bien the da thu nhung khong giu
- Noi rule on-route pickup thanh `detour_extra <= 2`: V8 tang nhung C6 va V6 tut nhieu, nen bo.
- Mo idle reposition xuong `N >= 11`: V2 tut manh, nen giu nguong `N >= 13`.

### Ghi chu chong overfit
- Cac thay doi duoc giu deu tang validation va khong lam tut public.
- Rule moi dua tren dac trung tong quat (`N`, `C`, detour rat nho), khong dua tren ten config.

## 2026-05-26 - Replace env structure from provided version

### Thay doi
- File: `env.py`
- Cap nhat cau truc `DeliveryEnv` theo ban env moi duoc cung cap:
  - dung `config_name`
  - tach tham so sinh don thanh `__lambda0`, `__surge_windows`, `__hotspots`, `__surge_amplitude`
  - `_order_rate()` nhan tham so rieng thay vi nhan ca `cfg`
  - `_init_shippers()` nhan `N`, `C`, `W_max`, `K_max`, `free_cells`
  - `load_config()` ho tro block `[SEED] base_seed = ...`
- File: `solvers/solver.py`
  - them fallback doc config tu env moi khi env khong co `public_cfg` hoac `cfg`.

### Ly do
Dong bo `env.py` voi ban moi duoc cung cap, dong thoi giu cac solver hien tai chay duoc voi API env moi.

### Benchmark
Lenh compile:

```powershell
python -m py_compile env.py run_test.py solvers\solver.py solvers\mapd_cbs_solver.py solvers\greedy_bfs.py solvers\vrp_ortools.py solvers\aco_solver.py
```

Lenh MAPD-CBS public:

```powershell
$env:PYTHONIOENCODING='utf-8'; python run_test.py --config test_config.txt --out results --method MAPDCBSSolver
```

Lenh MAPD-CBS validation:

```powershell
$env:PYTHONIOENCODING='utf-8'; python run_test.py --config validation_config.txt --out results_validation --method MAPDCBSSolver
```

Lenh all-method public:

```powershell
$env:PYTHONIOENCODING='utf-8'; python run_test.py --config test_config.txt --out results_all_after_env
```

### Ket qua

`MAPD-CBS` tren public sau khi doi env:

```text
Tong diem: 4144.1229
```

`MAPD-CBS` tren validation sau khi doi env:

```text
Tong diem: 5613.8469
```

All-method public:

| Method | Total score |
| --- | ---: |
| MAPD-CBS | 4144.12 |
| VRP-OrTools | 2934.99 |
| GreedyBFS | 2790.54 |
| ACO | 2790.54 |

### Ghi chu
- Diem `MAPD-CBS` khong doi so voi truoc khi thay env.
- Full benchmark cho thay cac solver khac van chay duoc.

## 2026-05-27 - Adaptive pickup memory and wider on-route pickup

### Thay doi
- File: `solvers/mapd_cbs_solver.py`
- Noi rule pickup gan nhu nam tren duong giao:
  - truoc: `N >= 19`, `via_pickup <= 1`, `detour_extra <= 1`
  - sau: `N >= 19`, `via_pickup <= 2`, `detour_extra <= 1`
  - van giu detour cuc nho de tranh lam tre don dang mang.
- Tang bo nho pickup gan day tu 10 len 20 diem.
- Lam bo nho idle reposition thich nghi theo so shipper:
  - neu `C <= 3`, chi dung 10 pickup moi nhat de tranh bi keo ve vung cu.
  - neu `C >= 4`, dung toi da 20 pickup moi nhat vi co nhieu shipper hon de phan tan.

### Ly do
Map lon co nhieu duong dai va vat can, nen pickup cach 2 buoc nhung gan nhu khong tang detour van dang de ghe. Voi idle reposition, validation cho thay:
- bo nho dai hon giup map co nhieu shipper nhu C4/V7 tang manh.
- bo nho qua dai lai lam map it shipper nhu V3 bi cham, nen can cua so gan hon khi `C <= 3`.

### Benchmark
Lenh public:

```powershell
$env:PYTHONIOENCODING='utf-8'; python run_test.py --config test_config.txt --out results --method MAPDCBSSolver
```

Lenh validation:

```powershell
$env:PYTHONIOENCODING='utf-8'; python run_test.py --config validation_config.txt --out results_validation --method MAPDCBSSolver
```

### Ket qua public

Tong diem `MAPD-CBS` tren `test_config.txt`:

```text
Truoc lan nay: 4144.1229
Sau lan nay:   4409.5283
Tang:          +265.4054
```

Chi tiet public sau toi uu:

| Config | Net reward | Delivered | On time | Late | Missed |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 247.6663 | 13/15 | 11 | 2 | 2 |
| C2 | 425.9418 | 23/25 | 20 | 3 | 2 |
| C3 | 487.1835 | 19/40 | 18 | 1 | 21 |
| C4 | 1028.4356 | 48/60 | 42 | 6 | 12 |
| C5 | 1037.9442 | 67/80 | 54 | 13 | 13 |
| C6 | 1182.3569 | 65/100 | 58 | 7 | 35 |

### Ket qua validation

Tong diem `MAPD-CBS` tren `validation_config.txt`:

```text
Truoc lan nay: 5613.8469
Sau lan nay:   5719.7176
Tang:          +105.8707
```

Chi tiet validation sau toi uu:

| Config | Net reward | Delivered | On time | Late | Missed |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1 | 382.2529 | 15/18 | 15 | 0 | 3 |
| V2 | 683.2727 | 30/32 | 29 | 1 | 2 |
| V3 | 701.0398 | 33/45 | 30 | 3 | 12 |
| V4 | 600.7921 | 40/65 | 29 | 11 | 25 |
| V5 | 642.9352 | 38/85 | 34 | 4 | 47 |
| V6 | 1356.3533 | 73/110 | 65 | 8 | 37 |
| V7 | 572.6223 | 36/55 | 30 | 6 | 19 |
| V8 | 780.4493 | 50/95 | 46 | 4 | 45 |

### Bien the da thu nhung khong giu
- Noi `via_pickup <= 3`: public khong doi nhieu nhung validation V6/V8 tut, nen bo.
- Doi route-aware coefficient map lon tu `0.5` sang `0.45`: validation co case tang nhung public C6 tut manh, nen bo.
- Bo nho pickup 30: diem bang 20, nen giu 20 de bot dung du lieu cu.

## 2026-05-27 - Solver compliance with no direct env/config reads

### Thay doi
- File: `solvers/solver.py`
  - Base `Solver` khong con doc `env.public_cfg`, `env.cfg`, `env.grid`, `env.N/C/G/T`.
  - Chi giu reference `env` de goi API hop le `reset()`, `step()`, `result()`.
- File: `solvers/mapd_cbs_solver.py`
  - Bo su dung `self.cfg` trong policy.
  - `grid`, `N`, `C` duoc lay tu `obs` hien tai:
    - `self.grid = obs["grid"]`
    - `self._map_size = obs["N"]`
    - `self._shipper_count = obs["C"]`
  - Cac rule toi uu tiep tuc dung thong tin nhin thay trong observation, khong doc surge/hotspot/config/hidden order.
- File: `solvers/greedy_bfs.py`, `solvers/vrp_ortools.py`, `solvers/aco_solver.py`
  - Sau `obs = self.env.reset()`, lay `grid/N/C/G/T` tu observation de full benchmark van chay duoc khi base Solver khong doc env internals.

### Ly do
Phu hop yeu cau moi: solver khong duoc doc truc tiep thong tin trong Env hoac doc file config de lay thong tin an phuc vu toi uu. Solver chi duoc dung observation hien tai.

### Kiem tra
Lenh grep:

```powershell
rg "self\\.cfg|env\\.cfg|public_cfg|env\\.grid|env\\.N|env\\.C|env\\.G|env\\.T|load_config|open\\(" -n solvers\\mapd_cbs_solver.py solvers\\solver.py
```

Ket qua chi con:

```text
solvers\solver.py:18:        self.cfg = {}
```

Lenh compile:

```powershell
python -m py_compile env.py run_test.py solvers\solver.py solvers\greedy_bfs.py solvers\vrp_ortools.py solvers\aco_solver.py solvers\mapd_cbs_solver.py
```

### Benchmark
MAPD-CBS public:

```powershell
$env:PYTHONIOENCODING='utf-8'; python run_test.py --config test_config.txt --out results --method MAPDCBSSolver
```

MAPD-CBS validation:

```powershell
$env:PYTHONIOENCODING='utf-8'; python run_test.py --config validation_config.txt --out results_validation --method MAPDCBSSolver
```

All-method public:

```powershell
$env:PYTHONIOENCODING='utf-8'; python run_test.py --config test_config.txt --out results_all_after_policy_fix
```

### Ket qua

`MAPD-CBS` sau khi bo doc env/config truc tiep:

```text
Public:     4409.5283
Validation: 5719.7176
```

All-method public:

| Method | Total score |
| --- | ---: |
| MAPD-CBS | 4409.53 |
| VRP-OrTools | 2934.99 |
| GreedyBFS | 2790.54 |
| ACO | 2790.54 |

### Ghi chu
- Diem MAPD-CBS khong doi sau khi lam sach nguon du lieu.
- Toi uu hien tai chi dung `obs`: map size, shipper count, grid, visible orders, visible shippers, new_order_ids.

## 2026-05-27 - Phase 2 path-cache optimization and stress benchmark

### Thay doi giu lai
- File: `solvers/mapd_cbs_solver.py`
  - Doi cache tim duong tu cache theo cap `(start, goal)` sang cache theo `start`.
  - Moi lan BFS tu mot vi tri se luu:
    - khoang cach tu `start` den tat ca o co the di;
    - buoc di dau tien tu `start` den tung o dich.
  - `_distance()` va `_next_move()` dung chung cache nay.
  - Cache duoc clear khi grid trong observation thay doi.
- File moi: `make_phase2_stress_config.py`
  - Tao `phase2_stress_config.txt` gom 8 config stress theo gioi han Phase 2:
    `N <= 100`, `C <= 25`, `G <= 1500`, `T <= 2400`.
  - Ban do connected, khong khai bao surge/hotspot trong config.

### Ly do
Phase 2 co the co ban do lon va nhieu don visible. Cach cu moi cap diem lai BFS rieng nen de bi nhan chi phi khi C/G/N tang. Cache theo nguon giup mot lan BFS tai vi tri shipper/order pickup co the phuc vu nhieu phep tinh khoang cach va next move.

### Ket qua giu lai

MAPD-CBS public:

```text
Total score: 4409.53
```

MAPD-CBS validation:

```text
Total score: 5719.72
```

Phase 2 stress:

```text
Total score: 19428.36
P2S8 N=100 C=25 G=1500 T=2400 chay xong trong 181.80s
Tong 8 config chay xong trong 471.1s
```

### Kiem tra

```powershell
python -m py_compile solvers\mapd_cbs_solver.py make_phase2_stress_config.py
$env:PYTHONIOENCODING='utf-8'; python run_test.py --config test_config.txt --out results_phase2_final_public --method MAPDCBSSolver
$env:PYTHONIOENCODING='utf-8'; python run_test.py --config validation_config.txt --out results_phase2_final_validation --method MAPDCBSSolver
$env:PYTHONIOENCODING='utf-8'; python run_test.py --config phase2_stress_config.txt --out results_phase2_stress --method MAPDCBSSolver
```

### Y tuong da thu nhung khong giu
- Thu them scoring theo `w/p/deadline` cho map lon `N >= 25` va mo rong recent pickup memory cho map lon.
- Ket qua stress giam tu `19428.36` xuong `15409.83`, nen da revert de tranh overfit va tranh lam solver tham don gia tri cao nhung xa qua.

### Ghi chu
- Toan bo policy van chi dung observation hien tai: grid, shippers, visible orders, new_order_ids, N/C/G/T trong obs.
- Khong doc env internals, khong doc config file, khong dung surge/hotspot hidden.

## 2026-05-27 - Restore main rolling-horizon CBS under compliance rules

### Thay doi
- File: `solvers/mapd_cbs_solver_exp.py`
  - Lay lai thuat toan manh tu nhanh main: rolling-horizon target assignment, CBS planning, fallback action, large-map mode, hotspot tracker online.
  - Bo hoan toan viec doc `env.public_cfg`, `env.N`, `env.C`, `env.G`, `env.T`.
  - Them `_refresh_from_obs(obs)` de cap nhat `grid/N/C/G/T` tu observation hien tai moi timestep.
  - Tat ca tinh reward/deadline/window/large-mode dung du lieu trong `obs`, khong dung surge/hotspot hidden.
- File: `solvers/mapd_cbs_solver.py`
  - Chuyen thanh wrapper cho `MAPDCBSSolver` compliant trong `mapd_cbs_solver_exp.py`.

### Ly do
Sau merge, neu giu solver compliant cu thi diem public chi dat `2515.71`. Solver main dat cao hon nhung vi pham rule do doc truc tiep env internals. Lan nay giu logic toi uu cua main, nhung doi nguon du lieu sang observation hop le.

### Kiem tra

```powershell
python -B -m py_compile solvers\mapd_cbs_solver.py solvers\mapd_cbs_solver_exp.py solvers\hotspot_tracker.py solvers\solver.py env.py run_test.py
$env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONIOENCODING='utf-8'; python -B run_test.py --config test_config.txt --out results_after_merge_restored_final --method MAPDCBSSolver
```

Grep compliance:

```powershell
rg "env\\.cfg|public_cfg|env\\.grid|env\\.N|env\\.C|env\\.G|env\\.T|load_config|open\\(" solvers\mapd_cbs_solver.py solvers\mapd_cbs_solver_exp.py solvers\solver.py
```

Khong co ket qua.

### Ket qua

```text
Sau merge truoc khi sua: 2515.71
Sau khi restore main logic compliant: 5713.45
Origin/main tren cung test_config sau merge: 5713.45
```

### Ghi chu
- Diem da ve lai muc main tren public config hien tai.
- Solver chinh van khong doc config file va khong doc thong tin an trong env.
