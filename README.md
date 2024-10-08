# Project Apero

Project Apero is a Streamlit application that replicates a simple chat interface similar to ChatGPT, using OpenAI's GPT-3.5-turbo model.

![Project Apero Banner](https://your-image-url.com/banner.png)

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [Running the Application](#running-the-application)
- [Notes](#notes)
- [License](#license)

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/hlexnc/project-apero.git
   cd project-apero
   ```

2. **Create a virtual environment and activate it:**

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Unix or macOS
   source venv/bin/activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your OpenAI API key:**

   - Rename `secrets.toml.example` to `secrets.toml` in the `.streamlit` directory.
   - Add your OpenAI API key to the `secrets.toml` file:

     ```toml
     OPENAI_API_KEY = "your-openai-api-key"
     ```

## Running the Application

Run the Streamlit app using the following command:

```bash
streamlit run app.py
```

## Notes

- Ensure you have an OpenAI API key. You can obtain one by signing up at [OpenAI](https://platform.openai.com/api-keys).
- The application uses the `gpt-3.5-turbo` model by default.
- Make sure not to commit your `secrets.toml` file or your API key to version control.

## License

This project is licensed under the MIT License.
