use burn::tensor::Tensor;

use super::ppo::Be;

pub fn eye(k: usize) -> Tensor<Be, 2> {
    let mut out = vec![];
    for i in 0..k {
        let mut row = vec![0.0f32; k];
        row[i] = 1.0f32;
        out.push(Tensor::from_floats(row.as_slice()).reshape([0, 1]));
    }
    Tensor::cat(out, 1)
}

pub fn diag(x: Tensor<Be, 1>) -> Tensor<Be, 2> {
    let dev = x.device();
    let [k] = x.shape().dims;
    x.unsqueeze() * eye(k).to_device(&dev)
}

pub fn cholesky(x: Tensor<Be, 2>) -> Tensor<Be, 2> {
    let dev = x.device();
    let [n, k] = x.shape().dims;
    assert_eq!(n, k);
    let mut l = Tensor::zeros_like(&x);
    for i in 0..n {
        for k in 0..i + 1 {
            let tmp_sum = if k == 0 {
                Tensor::zeros([1])
            } else {
                Tensor::cat(
                    (0..k)
                        .map(|j| {
                            l.clone().slice([i..i + 1, j..j + 1])
                                * l.clone().slice([k..k + 1, j..j + 1])
                        })
                        .collect(),
                    0,
                )
                .sum()
            }
            .unsqueeze()
            .to_device(&dev);
            if i == k {
                let values = (x.clone().slice([i..i + 1, i..i + 1]) - tmp_sum).sqrt();
                l = l.slice_assign([i..i + 1, k..k + 1], values);
            } else {
                let values = Tensor::ones([1, 1]).to_device(&dev)
                    / l.clone().slice([k..k + 1, k..k + 1])
                    * (x.clone().slice([i..i + 1, k..k + 1]) - tmp_sum);
                l = l.slice_assign([i..i + 1, k..k + 1], values);
            }
        }
    }
    l.transpose()
}

fn pivot(x: Tensor<Be, 2>) -> Tensor<Be, 2> {
    let [n, k] = x.shape().dims;
    assert_eq!(n, k);
    let mut id = eye(n);
    for i in 0..n {
        let mut max_pos = i;
        for j in i..n {
            if x.clone()
                .slice([i..i + 1, max_pos..max_pos + 1])
                .into_scalar()
                .abs()
                < x.clone().slice([i..i + 1, j..j + 1]).into_scalar().abs()
            {
                max_pos = j;
            }
        }

        if max_pos != i {
            let id_i = id.clone().slice([0..n, i..i + 1]);
            let id_max = id.clone().slice([0..n, max_pos..max_pos + 1]);
            id = id.slice_assign([0..n, max_pos..max_pos + 1], id_i);
            id = id.slice_assign([0..n, i..i + 1], id_max);
        }
    }
    id
}

pub fn lu(x: Tensor<Be, 2>) -> (Tensor<Be, 2>, Tensor<Be, 2>) {
    let [n, k] = x.shape().dims;
    assert_eq!(n, k);

    let mut l = eye(n);
    let mut u = Tensor::zeros_like(&x);

    let p = pivot(x.clone());
    let pa = p.matmul(x.clone());
    for j in 0..n {
        for i in 0..j + 1 {
            /*
            s1 = sum(U[k][j] * L[i][k] for k in xrange(i))
            U[i][j] = PA[i][j] - s1
             */
            let s1 = if i == 0 {
                Tensor::zeros([1])
            } else {
                Tensor::cat(
                    (0..i)
                        .map(|k| {
                            u.clone().slice([k..k + 1, j..j + 1])
                                * l.clone().slice([i..i + 1, k..k + 1])
                        })
                        .collect(),
                    0,
                )
                .sum()
            }
            .unsqueeze();
            let values = pa.clone().slice([i..i + 1, j..j + 1]) - s1;
            u = u.slice_assign([i..i + 1, j..j + 1], values);
        }

        for i in j..n {
            /*
            s2 = sum(U[k][j] * L[i][k] for k in xrange(j))
            L[i][j] = (PA[i][j] - s2) / U[j][j]
             */
            let s2 = if j == 0 {
                Tensor::zeros([1])
            } else {
                Tensor::cat(
                    (0..j)
                        .map(|k| {
                            u.clone().slice([k..k + 1, j..j + 1])
                                * l.clone().slice([i..i + 1, k..k + 1])
                        })
                        .collect(),
                    0,
                )
                .sum()
            }
            .unsqueeze();
            let values = (pa.clone().slice([i..i + 1, j..j + 1]) - s2)
                / u.clone().slice([j..j + 1, j..j + 1]);
            l = l.slice_assign([i..i + 1, j..j + 1], values);
        }
    }

    (l.transpose(), u.transpose())
}

// https://rosettacode.org/wiki/Gauss-Jordan_matrix_inversion#Rust
pub fn inverse(x: Tensor<Be, 2>) -> Tensor<Be, 2> {
    let dev = x.device();
    let [n, k] = x.shape().dims;
    assert_eq!(n, k);
    let diag = x.clone().matmul(eye(n).to_device(&dev));
    for i in diag.into_data().value {
        assert_ne!(i, 0.0);
    }

    let aug = vec![x.clone(), eye(n)];
    let mut aug = Tensor::cat(aug, 1);
    print_matrix(&aug);
    // dbg!(aug.shape().dims);

    // gauss-jordan generalized reduced row echelon form
    let mut lead = 0;
    let [nrows, ncols] = aug.shape().dims;

    'outer: for r in 0..nrows {
        if ncols <= lead {
            break;
        }
        let mut i = r;
        let mut aug_i_lead = aug.clone().slice([i..i + 1, lead..lead + 1]).into_scalar();
        while aug_i_lead == 0.0 {
            i += 1;
            if i == nrows {
                i = r;
                lead += 1;
                if lead == ncols {
                    break 'outer;
                }
            }
            aug_i_lead = aug.clone().slice([i..i + 1, lead..lead + 1]).into_scalar();
        }

        let temp = aug.clone().slice([i..i + 1, 0..ncols]);
        aug = aug.clone().slice_assign(
            [i..i + 1, 0..ncols],
            aug.clone().slice([r..r + 1, 0..ncols]),
        );
        aug = aug.clone().slice_assign([r..r + 1, 0..ncols], temp);

        // print_matrix(&aug);
        let div = aug.clone().slice([r..r + 1, lead..lead + 1]).into_scalar();
        // dbg!(div);
        if div != 0.0 {
            for j in 0..ncols {
                aug = aug.clone().slice_assign(
                    [r..r + 1, j..j + 1],
                    aug.clone().slice([r..r + 1, j..j + 1]) / div,
                );
            }
        }
        for k in 0..nrows {
            if k != r {
                let mult = aug.clone().slice([k..k + 1, lead..lead + 1]).into_scalar();
                // dbg!(mult);
                for j in 0..ncols {
                    aug = aug.clone().slice_assign(
                        [k..k + 1, j..j + 1],
                        aug.clone().slice([k..k + 1, j..j + 1])
                            - aug.clone().slice([r..r + 1, j..j + 1]) * mult,
                    );
                }
            }
        }
        lead += 1;
    }
    print_matrix(&aug);

    aug.slice([0..n, n..n * 2])
}

fn print_matrix(x: &Tensor<Be, 2>) {
    let [rows, cols] = x.shape().dims;
    for i in 0..rows {
        println!("{:?}", x.clone().slice([i..i + 1, 0..cols]).to_data().value);
    }
    println!();
}

#[cfg(test)]
mod tests {
    use burn_tensor::Tensor;

    use super::{cholesky, diag, eye, inverse, lu, print_matrix, Be};

    #[test]
    fn test_diag() {
        let e = eye(4);
        for i in 0..4 {
            assert_eq!(e.clone().slice([i..i + 1, i..i + 1]).into_scalar(), 1.0f32);
        }
        let x = Tensor::from_floats([0.0f32, 1.0, 2.0, 3.0]);
        let x = diag(x);
        for i in 0..4 {
            assert_eq!(
                x.clone().slice([i..i + 1, i..i + 1]).into_scalar(),
                i as f32
            );
        }
    }

    fn assert_all_close_4x4(a: Tensor<Be, 2>, b: nalgebra::DMatrix<f32>) {
        for i in 0..16 {
            let i1 = b.data.as_slice()[i];
            let i2 = a.to_data().value[i];
            // println!("{}, {}: i1={}, i2={}", i % 4, i / 4, i1, i2);
            assert!(
                (i1 - i2).abs() < 1e-5,
                "{}, {}: i1={}, i2={}",
                i % 4,
                i / 4,
                i1,
                i2
            );
        }
    }

    #[test]
    fn test_choelsky() {
        #[rustfmt::skip]
        let x: [f32; 16] = [
            6., 3., 4., 8.,
            3., 6., 5., 1.,
            4., 5., 10., 7.,
            8., 1., 7., 25.,
        ];
        let x_nalg = nalgebra::DMatrix::from_row_slice(4, 4, &x);
        let x_tensor = Tensor::<Be, 1>::from_floats(x).reshape([4, 4]);
        for i in 0..16 {
            let i1 = x_nalg.data.as_slice()[i];
            let i2 = x_tensor.to_data().value[i];
            assert_eq!(i1, i2);
        }

        let c_nalg = x_nalg.cholesky().unwrap().unpack();
        let c_tensor = cholesky(x_tensor);
        assert_all_close_4x4(c_tensor, c_nalg);
    }

    #[test]
    fn test_lu() {
        #[rustfmt::skip]
        let x: [f32; 16] = [
            7., 3., -1., 2., 
            3., 8., 1., -4., 
            -1., 1., 4., -1., 
            2., -4., -1., 3., 
        ];
        let x_nalg = nalgebra::DMatrix::from_row_slice(4, 4, &x);
        let x_tensor = Tensor::<Be, 1>::from_floats(x).reshape([4, 4]);
        for i in 0..16 {
            let i1 = x_nalg.data.as_slice()[i];
            let i2 = x_tensor.to_data().value[i];
            assert_eq!(i1, i2);
        }

        let lu_nalg = x_nalg.lu();
        let (l_tensor, u_tensor) = lu(x_tensor.clone());
        assert_all_close_4x4(l_tensor, lu_nalg.l());
        assert_all_close_4x4(u_tensor, lu_nalg.u());

        let x_tensor = Tensor::from_floats([1., 2., 3., 4., 1., 6., 7., 8., 9.]).reshape([3, 3]);
        print_matrix(&x_tensor);
        let x_inv = inverse(x_tensor.clone());
        // dbg!(x_inv.to_data());
        print_matrix(&x_inv);
        let i = x_inv.matmul(x_tensor);
        print_matrix(&i);
    }
}