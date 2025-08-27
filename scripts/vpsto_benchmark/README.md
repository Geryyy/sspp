```bash
python3 benchmark_vpsto.py \
  --xml /path/to/your_model.xml \
  --hooks ./hooks.py \
  --ndof 4 \
  --q0 0.1,0.0,0.4,0.0 \
  --qT 0.6,0.2,0.3,0.0 \
  --N 50 \
  --N_via 1 \
  --N_eval 50 \
  --pop_size 100 \
  --max_iter 60 \
  --sigma_init 4.0 \
  --lam_coll 1e3
```


```bash
python3 benchmark_vpsto.py \
  --xml /home/gebmer/repos/robocrane/sspp/mjcf/robocrane/robocrane.xml \
  --hooks ./hooks.py \
  --body gripper_collision_with_block/ \
  --ndof 4 \
  --q0 0.5,0.15,0.116,1.57 \
  --qT 0.5,-0.05,0.116,0.00 \
  --N 50 \
  --N_via 3 \
  --N_eval 50 \
  --pop_size 100 \
  --max_iter 60 \
  --sigma_init 4.0 \
  --lam_coll 1e3 \
  --preview \
  --preview_iter 40 \
  --preview_fps 60

```


no preview 
```bash
python3 benchmark_vpsto.py \
  --xml /home/gebmer/repos/robocrane/sspp/mjcf/robocrane/robocrane.xml \
  --hooks ./hooks.py \
  --body gripper_collision_with_block/ \
  --ndof 4 \
  --q0 0.5,0.15,0.116,1.57 \
  --qT 0.5,-0.05,0.116,0.00 \
  --N 50 \
  --N_via 3 \
  --N_eval 50 \
  --pop_size 100 \
  --max_iter 60 \
  --sigma_init 4.0 \
  --lam_coll 1e3 
```

## icra paper benchmark call


python3 icra_bench_vpsto.py \
  --xml /home/geraldebmer/repos/sspp/mjcf/robocrane/robocrane.xml \
  --hooks ./hooks.py \
  --ndof 4 \
  --q0 0.5,0.15,0.116,1.57 \
  --qT 0.5,-0.05,0.116,0.00 \
  --N 50 \
  --N_via 3 \
  --N_eval 50 \
  --pop_size 100 \
  --max_iter 60 \
  --sigma_init 4.0 \
  --lam_coll 1e3


If you also want anytime mode with budgets, just add e.g.:
  --budgets_ms "20,50,100" --chunk_iter 5
