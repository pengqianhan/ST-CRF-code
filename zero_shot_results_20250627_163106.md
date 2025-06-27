# Zero-Shot Trajectory Prediction Results

**Generated on:** 2025-06-27 16:31:06  
**Total Combinations Evaluated:** 25  
**KSTEPS:** 1  
**ROBUSTNESS:** 0  

## Summary Table

| Training Dataset | Testing Dataset | ADE | FDE | Status |
|------------------|-----------------|-----|-----|--------|
| eth | hotel | 0.391 | 0.7429 | ✅ Success |
| eth | univ | 0.6182 | 1.2589 | ✅ Success |
| eth | zara1 | 0.4745 | 0.99 | ✅ Success |
| eth | zara2 | 0.4411 | 0.8864 | ✅ Success |
| eth | sdd | 0.7516 | 1.4708 | ✅ Success |
| hotel | eth | 0.8748 | 1.5514 | ✅ Success |
| hotel | univ | 0.5878 | 1.2126 | ✅ Success |
| hotel | zara1 | 0.5827 | 1.1535 | ✅ Success |
| hotel | zara2 | 0.4647 | 0.9489 | ✅ Success |
| hotel | sdd | 0.877 | 1.6902 | ✅ Success |
| univ | eth | 0.9449 | 2.0231 | ✅ Success |
| univ | hotel | 0.4332 | 0.7722 | ✅ Success |
| univ | zara1 | 0.5061 | 1.0523 | ✅ Success |
| univ | zara2 | 0.3808 | 0.7957 | ✅ Success |
| univ | sdd | 0.8414 | 1.6133 | ✅ Success |
| zara1 | eth | 0.7941 | 1.7392 | ✅ Success |
| zara1 | hotel | 0.4216 | 0.7975 | ✅ Success |
| zara1 | univ | 0.5445 | 1.1461 | ✅ Success |
| zara1 | zara2 | 0.3426 | 0.7242 | ✅ Success |
| zara1 | sdd | 0.7017 | 1.3692 | ✅ Success |
| zara2 | eth | 0.8821 | 1.8679 | ✅ Success |
| zara2 | hotel | 0.5047 | 0.9253 | ✅ Success |
| zara2 | univ | 0.5693 | 1.1728 | ✅ Success |
| zara2 | zara1 | 0.5141 | 1.0403 | ✅ Success |
| zara2 | sdd | 0.8595 | 1.6399 | ✅ Success |

## Detailed Results

### Model trained on ETH

- **HOTEL**: ADE=0.391, FDE=0.7429
- **UNIV**: ADE=0.6182, FDE=1.2589
- **ZARA1**: ADE=0.4745, FDE=0.99
- **ZARA2**: ADE=0.4411, FDE=0.8864
- **SDD**: ADE=0.7516, FDE=1.4708

### Model trained on HOTEL

- **ETH**: ADE=0.8748, FDE=1.5514
- **UNIV**: ADE=0.5878, FDE=1.2126
- **ZARA1**: ADE=0.5827, FDE=1.1535
- **ZARA2**: ADE=0.4647, FDE=0.9489
- **SDD**: ADE=0.877, FDE=1.6902

### Model trained on UNIV

- **ETH**: ADE=0.9449, FDE=2.0231
- **HOTEL**: ADE=0.4332, FDE=0.7722
- **ZARA1**: ADE=0.5061, FDE=1.0523
- **ZARA2**: ADE=0.3808, FDE=0.7957
- **SDD**: ADE=0.8414, FDE=1.6133

### Model trained on ZARA1

- **ETH**: ADE=0.7941, FDE=1.7392
- **HOTEL**: ADE=0.4216, FDE=0.7975
- **UNIV**: ADE=0.5445, FDE=1.1461
- **ZARA2**: ADE=0.3426, FDE=0.7242
- **SDD**: ADE=0.7017, FDE=1.3692

### Model trained on ZARA2

- **ETH**: ADE=0.8821, FDE=1.8679
- **HOTEL**: ADE=0.5047, FDE=0.9253
- **UNIV**: ADE=0.5693, FDE=1.1728
- **ZARA1**: ADE=0.5141, FDE=1.0403
- **SDD**: ADE=0.8595, FDE=1.6399

## Statistics

- **Successful Evaluations:** 25/25
- **Mean ADE:** 0.6122
- **Mean FDE:** 1.2234
- **Min ADE:** 0.3426
- **Max ADE:** 0.9449
- **Min FDE:** 0.7242
- **Max FDE:** 2.0231

## Best Performing Combinations

### Best ADE
- **zara1 → zara2**: ADE=0.3426, FDE=0.7242

### Best FDE
- **zara1 → zara2**: ADE=0.3426, FDE=0.7242

