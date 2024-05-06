# Adult
for seed in 0 1 2 3 4
do
    python tabular_erm.py --dataset adult --sensitive_attr sex --target_attr income --seed $seed
    for lam in 0.2 0.4 0.6 0.8 1.0
    do
        python tabular_advdebias.py --dataset adult --sensitive_attr sex --target_attr income --lam $lam --seed $seed
    done
    for lam in 0.1 0.2 0.3 0.4 0.5
    do
        python tabular_diffdp.py --dataset adult --sensitive_attr sex --target_attr income --lam $lam --seed $seed
        python tabular_mixup.py --dataset adult --sensitive_attr sex --target_attr income --lam $lam --seed $seed
        python tabular_dralign.py --dataset adult --sensitive_attr sex --target_attr income --lam $lam --seed $seed
        python tabular_diffabcc.py --dataset adult --sensitive_attr sex --target_attr income --lam $lam --seed $seed
    done
    for lam in 0.1 0.15 0.2 0.25 0.3
    do
        python tabular_diffmcdp.py --dataset adult --sensitive_attr sex --target_attr income --lam $lam --seed $seed
    done
done


# Bank
for seed in 0 1 2 3 4
do
    python tabular_erm.py --dataset bank_marketing --sensitive_attr age --target_attr credit --seed $seed
    for lam in 0.3 0.6 0.9 1.2 1.5
    do
        python tabular_advdebias.py --dataset bank_marketing --sensitive_attr age --target_attr credit --lam $lam --seed $seed
    done
    for lam in 0.1 0.2 0.3 0.4 0.5
    do
        python tabular_diffdp.py --dataset bank_marketing --sensitive_attr age --target_attr credit --lam $lam --seed $seed
        python tabular_dralign.py --dataset bank_marketing --sensitive_attr age --target_attr credit --lam $lam --seed $seed
        python tabular_diffabcc.py --dataset bank_marketing --sensitive_attr age --target_attr credit --lam $lam --seed $seed
    done
    for lam in 0.3 0.4 0.5 0.6 0.7
    do
        python tabular_mixup.py --dataset bank_marketing --sensitive_attr age --target_attr credit --lam $lam --seed $seed
    done
    for lam in 0.1 0.15 0.2 0.25 0.3
    do
        python tabular_diffmcdp.py --dataset bank_marketing --sensitive_attr age --target_attr credit --lam $lam --seed $seed
    done
done


# CelebA-A
for seed in 0 1 2 3 4
do
    python image_erm.py --sensitive_attr Gender --target_attr Attractive --seed $seed
    for lam in 0.5 1 1.5 2 2.5
    do
        python image_advdebias.py --sensitive_attr Gender --target_attr Attractive --lam $lam --seed $seed
    done
    for lam in 0.1 0.2 0.3 0.4 0.5
    do
        python image_diffdp.py --sensitive_attr Gender --target_attr Attractive --lam $lam --seed $seed
        python image_dralign.py --sensitive_attr Gender --target_attr Attractive --lam $lam --seed $seed
        python image_diffabcc.py --sensitive_attr Gender --target_attr Attractive --lam $lam --seed $seed
    done
    for lam in 0.5 1 2 3 4
    do
        python image_mixup.py --sensitive_attr Gender --target_attr Attractive --lam $lam --seed $seed
    done
    for lam in 0.1 0.15 0.2 0.25 0.3
    do
        python image_diffmcdp.py --sensitive_attr Gender --target_attr Attractive --lam $lam --seed $seed
    done
done


# CelebA-W
for seed in 0 1 2 3 4
do
    python image_erm.py --sensitive_attr Gender --target_attr Wavy_Hair --seed $seed
    for lam in 0.5 1 1.5 2 2.5
    do
        python image_advdebias.py --sensitive_attr Gender --target_attr Wavy_Hair --lam $lam --seed $seed
    done
    for lam in 0.1 0.2 0.3 0.4 0.5
    do
        python image_diffdp.py --sensitive_attr Gender --target_attr Wavy_Hair --lam $lam --seed $seed
        python image_dralign.py --sensitive_attr Gender --target_attr Wavy_Hair --lam $lam --seed $seed
        python image_diffabcc.py --sensitive_attr Gender --target_attr Wavy_Hair --lam $lam --seed $seed
    done
    for lam in 0.5 1 1.5 2 2.5
    do
        python image_mixup.py --sensitive_attr Gender --target_attr Wavy_Hair --lam $lam --seed $seed
    done
    for lam in 0.1 0.15 0.2 0.25 0.3
    do
        python image_diffmcdp.py --sensitive_attr Gender --target_attr Wavy_Hair --lam $lam --seed $seed
    done
done