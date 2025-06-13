<<<<<<< HEAD
# HR CV Filtering System with LLM Integration

## Overview

This application is designed to help Human Resources (HR) professionals streamline the recruitment process. It uses Hugging Face's DistilBERT (a powerful LLM model) to match uploaded CVs against a given job description. The system ranks CVs based on similarity to the job description and allows HR to filter and download the top qualifying candidates.

## Features

- **Job Description Input**: HR can paste a job description and have it analyzed by the system.
- **Multiple CV Uploads**: Allows HR to upload multiple CVs in PDF, DOCX, or TXT format.
- **Ranking and Filtering**: The system ranks the CVs based on their similarity to the job description, with an option to filter and display the top N candidates.
- **Download Option**: HR can download the top CVs for further review.
- **Similarity Scoring**: Displays a similarity percentage for each CV, indicating how well the CV matches the job description.

## Requirements

To run the project, make sure you have the following installed:

- Python 3.x
- Streamlit
- Hugging Face Transformers
- PyPDF2
- Docx
- Scikit-learn
- Torch

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/HR_CV_Filtering_App.git
    ```

2. Navigate into the project directory:

    ```bash
    cd HR_CV_Filtering_App
    ```

3. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    source venv/bin/activate  # macOS/Linux
    ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the application:

    ```bash
    streamlit run cv_filtering.py
    ```

2. Enter the job description in the provided text box.
3. Upload the CV files (PDF, DOCX, or TXT).
4. Click on the "Filter and Rank Candidates" button to see the results.
5. Use the sidebar to filter the top N candidates.
6. Download the selected CVs.

## Contributing

If you want to contribute to this project, feel free to fork the repository and create a pull request with your changes. We welcome contributions to improve the functionality, features, or design of this app!

## License

This project is open source and available under the [MIT License](LICENSE).
=======
# HR_CV_FILTERING_APP
This application is designed to streamline the recruitment process by efficiently matching CVs to job descriptions. Using Natural Language Processing (NLP) with the power of Hugging Face's DistilBERT model, the system evaluates and ranks candidates based on how closely their CV matches a provided job description.
>>>>>>> a66aec3c6d819cb0a34395a13f34a5f41e11ed8f
