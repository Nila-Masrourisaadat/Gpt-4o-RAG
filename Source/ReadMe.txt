How to Execute the Code:

Prerequisites:

Python 3.8 or higher
Azure account (for Blob Storage)
OpenAI API key
Virtual environment (recommended)
Installation

Clone the Repository:

git clone 'link to Github repo'
cd your-repo-name

Set Up a Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Requirements:

pip install -r requirements.txt

Place All Python Files in One Directory:

Ensure that rag.py, rag_llm_wrapper.py, config.py, and rag_text_helper.py are all in the same directory.

Configuration:
Set Up Environment Variables in the rag.py script

Execution:
Run the Main Script:

python rag.py

Interacting with the System:

You will be prompted to enter your questions.
The system will process the documents and provide answers based on the retrieved information.
Type 'done' to exit the interactive session.

License:
This project is licensed under the Apache License 2.0. See the LICENSE file for details.

# SPDX-FileCopyrightText: 2023-present Oori Data <info@oori.dev>
# SPDX-License-Identifier: Apache-2.0