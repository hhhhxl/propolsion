# srm_computing

## Step1: Clone Repository

```bash
git clone https://github.com/hhhhxl/propolsion.git
cd propolsion
```

## Step2: Create Conda Evironment
```bash
conda create -n srm python=3.10
conda activate srm
```

### Note: Please install Anaconda！！

## Step3: Install Requirements

```bash
pip install -r requirements.txt
```

## Step4: Run!

`python main.py`




# 模块开发进度

- [x] 输出纯气相P-T曲线
- [x] 输出两相流P-T曲线
- [x] 输出F-T曲线
- [x] 计算点火药量
- [ ] 用户交互
- [x] 总冲比冲计算

---

# note: 

目前版本肉厚曲线单位为英寸和英寸^2

At单位为cm2

密度g/cm3

装药体积cm3

"rho_s": 凝聚相密度，单位为g/cm3

"dc0": 凝聚相参考粒子直径，单位为um

"ep": 凝聚相质量比

"phi_a": 侵蚀函数，此版本未开发，默认为1

# Update Date：2024.8.11 更新
