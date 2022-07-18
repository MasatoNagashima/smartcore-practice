use smartcore::dataset::iris::load_dataset;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use smartcore::model_selection::train_test_split;
use smartcore::metrics::accuracy;

fn main() {
    let irirs_data = load_dataset();
    let x = DenseMatrix::from_array(
        irirs_data.num_samples,
        irirs_data.num_features,
        &irirs_data.data,
    );
    let y = irirs_data.target;
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, true);

    let knn = KNNClassifier::fit(
        &x_train,
        &y_train,
        Default::default(),
    ).unwrap();
    let y_pred = knn.predict(&x_test).unwrap();
    println!("accuracy: {}", accuracy(&y_test, &y_pred));
}
