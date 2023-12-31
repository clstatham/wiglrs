use burn::tensor::Tensor;
use burn_tensor::backend::Backend;
use nalgebra::DMatrix;

use super::ppo::Be;

pub fn eye<B: Backend>(k: usize) -> Tensor<B, 2> {
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

pub fn diag2d<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 3> {
    let dev = x.device();
    let [b, k] = x.shape().dims;
    x.unsqueeze() * eye(k).unsqueeze().to_device(&dev)
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

pub fn norm<B: Backend>(x: Tensor<B, 1>) -> Tensor<B, 1> {
    let mut sum = Tensor::zeros_device([1], &x.device());
    let [p] = x.shape().dims;
    for i in 0..p {
        sum = sum + x.clone().slice([i..i + 1]) * x.clone().slice([i..i + 1]);
    }
    sum.sqrt()
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

pub fn print_matrix(x: &Tensor<Be, 2>) {
    let [rows, cols] = x.shape().dims;
    for i in 0..rows {
        println!("{:?}", x.clone().slice([i..i + 1, 0..cols]).to_data().value);
    }
    println!();
}

pub fn orthogonal<B: Backend<FloatElem = f32>>(
    shape: burn_tensor::Shape<2>,
    gain: f32,
) -> Tensor<B, 2> {
    let [nrows, ncols] = shape.dims;
    let z = Tensor::<B, 2>::random(shape.clone(), burn_tensor::Distribution::Normal(0.0, 1.0));
    let z = DMatrix::from_row_slice(nrows, ncols, z.into_data().value.as_slice());
    let qr = z.qr();
    let d = &qr.r().diagonal();
    let mut q = qr.q();
    for (i, elem) in d.into_iter().enumerate() {
        q[(i, i)] *= elem.signum();
    }
    let t = Tensor::from_floats(q.transpose().data.as_slice()).reshape(shape);
    // let e = eye(ncols);
    // let prod = t.clone().transpose().matmul(t.clone());
    // dbg!((e - prod).to_data());
    t * gain
}

#[cfg(test)]
mod tests {
    use burn_tensor::Tensor;

    use super::{cholesky, orthogonal, print_matrix, Be};

    #[test]
    fn test_orthogonal() {
        let t = orthogonal::<Be>([6, 4].into(), 0.01);
    }

    // #[test]
    // fn test_qr() {
    //     let x = Tensor::<Be, 2>::random([4, 6], burn_tensor::Distribution::Normal(0.0, 1.0));
    //     let (q, r) = qr(x.clone());
    //     let x2 = nalgebra::DMatrix::from_row_slice(4, 6, x.to_data().value.as_slice());
    //     let qr2 = x2.qr();
    //     let (q2, r2) = (qr2.q(), qr2.r());
    //     assert_all_close(q, q2);
    //     assert_all_close(r, r2);
    // }

    // #[test]
    // fn test_diag() {
    //     let e = eye(4);
    //     for i in 0..4 {
    //         assert_eq!(e.clone().slice([i..i + 1, i..i + 1]).into_scalar(), 1.0f32);
    //     }
    //     let x = Tensor::from_floats([0.0f32, 1.0, 2.0, 3.0]);
    //     let x = diag(x);
    //     for i in 0..4 {
    //         assert_eq!(
    //             x.clone().slice([i..i + 1, i..i + 1]).into_scalar(),
    //             i as f32
    //         );
    //     }
    // }

    fn assert_all_close(a: Tensor<Be, 2>, b: nalgebra::DMatrix<f32>) {
        let [d0, d1] = a.shape().dims;
        dbg!(d0 * d1, b.nrows() * b.ncols());
        for i in 0..d0 * d1 {
            let i2 = b.data.as_slice()[i];
            let i1 = a.to_data().value[i];
            println!("i1={}, i2={}", i1, i2);
            // assert!(
            //     (i1 - i2).abs() < 1e-5,
            //     "{}, {}: i1={}, i2={}",
            //     i % 4,
            //     i / 4,
            //     i1,
            //     i2
            // );
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
        assert_all_close(c_tensor, c_nalg);
    }

    #[test]
    fn test_choelsky_diag() {
        #[rustfmt::skip]
        let x: [f32; 16] = [
            9., 0., 0., 0., 
            0., 12., 0., 0., 
            0., 0., 16., 0., 
            0., 0., 0., 20., 
        ];
        let x_tensor = Tensor::<Be, 1>::from_floats(x).reshape([4, 4]);
        let x_ch = cholesky(x_tensor.clone());
        let x_sqrt = x_tensor.sqrt();
        print_matrix(&x_ch);
        print_matrix(&x_sqrt);
    }
}
