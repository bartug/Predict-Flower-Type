# Flower Classification and Care Recommendation

This project aims to classify flowers from images and provide care recommendations in Turkish based on the predicted flower type. We use a convolutional neural network (CNN) for image classification and a CSV file for flower care information.

## Dataset

We use the [Extended Flowers Recognition](https://www.kaggle.com/datasets/jonathanflorez/extended-flowers-recognition?resource=download) dataset from Kaggle, which contains images of 10 different flower types: Aster, Daffodil, Dahlia, Daisy, Dandelion, Iris, Orchid, Rose, Sunflower, and Tulip.

## Project Structure

- **flowers/**: Directory containing images of the flowers organized by their respective categories.
- **flower_care_tr.csv**: CSV file containing flower care information in Turkish.
- **flower_classification.py**: Python script for training the model and making predictions.
- **requirements.txt**: List of required Python packages.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/flower-classification-and-care.git
    cd flower-classification-and-care
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/jonathanflorez/extended-flowers-recognition?resource=download) and place the `flowers` directory in the root of your project.

## Usage

1. **Training the Model**: Run the following command to train the model.

    ```bash
    python flower_classification.py --train
    ```

2. **Making Predictions**: To predict the flower type from an image and get care recommendations, use the following command:

    ```bash
    python flower_classification.py --predict --image path_to_image.jpg
    ```

    The script will output the predicted flower type and care instructions in Turkish.

## Flower Care Information

The `flower_care_tr.csv` file contains the following columns:

- **Çiçek Türü**: Flower type in Turkish.
- **Sulama Sıklığı**: Watering frequency.
- **Nem Seviyesi**: Humidity level.
- **Toprak Tipi**: Soil type.
- **Işık İhtiyacı**: Light requirement.

Example entry:

| Flower Type | Watering Frequency    | Humidity Level | Soil Type        | Light Requirement      | Çiçek Türü  |
|-------------|-----------------------|----------------|------------------|------------------------|-------------|
| Rose        | Weekly                | Moderate       | Well-drained     | Full sun to partial shade | Gül         |
| Sunflower   | Twice a week          | Low            | Loamy            | Full sun               | Ayçiçeği    |
| Daffodil    | Once every two weeks  | Moderate       | Well-drained     | Full sun to partial shade | Nergis      |

## Example

When the model predicts a flower as a "Rose", the script will output:

Predicted Flower Type: Rose
Care Instructions: Güller genellikle haftada bir kez sulanmalı ve orta derecede nemli toprak tercih eder.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Thanks to Kaggle for providing the [Extended Flowers Recognition](https://www.kaggle.com/datasets/jonathanflorez/extended-flowers-recognition?resource=download) dataset.
