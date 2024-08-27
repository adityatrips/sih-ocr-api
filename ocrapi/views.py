import json
import cv2
import numpy as np
import time
from rest_framework.views import APIView
from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework import status
from rapidocr_onnxruntime import RapidOCR
from pdf2image import convert_from_bytes
from groq import Groq
from django.conf import settings


class ImageOCR(APIView):
    parser_class = (FileUploadParser,)
    groq = Groq(
        api_key=settings.GROQ_API_KEY,
    )

    def post(self, request, format=None):
        file_obj = request.data.get("file")

        if file_obj is None:
            return Response(
                {"error": "No file was uploaded"}, status=status.HTTP_400_BAD_REQUEST
            )

        if not file_obj.content_type.startswith("image/"):
            return Response(
                {"error": "Invalid file type"}, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            start_time = time.time()

            image = cv2.imdecode(
                np.frombuffer(file_obj.read(), np.uint8), cv2.IMREAD_UNCHANGED
            )

            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            image = cv2.filter2D(image, -1, kernel)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ocr = RapidOCR()
            result = list(ocr(gray))

            del result[1]
            for item in result:
                for subitem in item:
                    del subitem[0]

            for item in result:
                for subitem in item:
                    subitem[0] = subitem[0].replace('"', "")
                    del subitem[1]

            formatted_result = []

            for item in result:
                for subitem in item:
                    formatted_result.append(subitem[0])

            end_time = time.time()

            chat_completion = self.groq.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"""
Extract the following information from the provided document and return it in JSON format:

- `name`: The name of the person
- `dob`: The date of birth of the person in DD-MM-YYYY format
- `phone`: The phone number of the person. It should be a ten-digit number. If uncertain, return null. The format follows: `b(?:\+91\s?)?\(?(\d{10})\)?`
- `document_number`: The document number, which could be:
  - A 12-digit Aadhaar card number (format: `[\d]{4} [\d]{4} [\d]{4}`)
  - A 10-digit PAN card number (format: `[A-Z]{5}[\d]{4}[A-Z]`)
  - A driving license number (format: `[A-Za-z]{2}[\d]{2}[\d]{11}`)
  - Or other types of document numbers
- `address`: The address of the person. If uncertain, return null
- `type`: The type of document (e.g., Aadhaar card, marksheet)
- `language`: The language in which the text is written. If uncertain, return null
- `gender`: The gender of the person. If uncertain, return null

Ensure the following:
- Only the specified information is included in the output.
- The output is a single JSON object.
- The JSON response starts with `{{` and ends with `}}`.
- No additional comments, notes, or extra information are included.
- Each document will have only one `document_number` key.

Output only the JSON data as described above.

Document:
{formatted_result}
""",
                    }
                ],
                model="llama-3.1-70b-versatile",
            )

            print(chat_completion.choices[0].message)

            return Response(
                {
                    "execution_time": end_time - start_time,
                    "ocr_result": json.loads(
                        "{"
                        + chat_completion.choices[0]
                        .message.content.strip()
                        .split("{")[1]
                        .split("}")[0]
                        + "}"
                    ),
                }
            )
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class PdfOCR(APIView):
    parser_class = (FileUploadParser,)
    groq = Groq(
        api_key=settings.GROQ_API_KEY,
    )

    def post(self, request, format=None):
        file_obj = request.data.get("file")

        if file_obj is None:
            return Response(
                {"error": "No file was uploaded"}, status=status.HTTP_400_BAD_REQUEST
            )

        if file_obj.content_type != "application/pdf":
            return Response(
                {"error": "Invalid file type"}, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            start_time = time.time()

            # Convert PDF to images
            pdf_file = file_obj.read()
            images = convert_from_bytes(pdf_file, dpi=350)

            # Process only the first page
            image = images[0]
            image = np.array(image)

            # Apply filter for image enhancement
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            image = cv2.filter2D(image, -1, kernel)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Perform OCR
            ocr = RapidOCR()
            result = list(ocr(gray))

            # Processing result to format it similarly to ImageOCR
            del result[1]
            for item in result:
                for subitem in item:
                    del subitem[0]

            formatted_result = []

            for item in result:
                for subitem in item:
                    subitem[0] = subitem[0].replace('"', "")
                    del subitem[1]
                    formatted_result.append(subitem[0])

            # Send formatted result for further processing with Groq
            chat_completion = self.groq.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"""
Extract the following information from the provided document and return it in JSON format:

- `name`: The name of the person
- `dob`: The date of birth of the person in DD-MM-YYYY format
- `phone`: The phone number of the person. It should be a ten-digit number. If uncertain, return null. The format follows: `b(?:\+91\s?)?\(?(\d{10})\)?`
- `document_number`: The document number, which could be:
  - A 12-digit Aadhaar card number (format: `[\d]{4} [\d]{4} [\d]{4}`)
  - A 10-digit PAN card number (format: `[A-Z]{5}[\d]{4}[A-Z]`)
  - A driving license number (format: `[A-Za-z]{2}[\d]{2}[\d]{11}`)
  - Or other types of document numbers
- `address`: The address of the person. If uncertain, return null
- `type`: The type of document (e.g., Aadhaar card, marksheet)
- `language`: The language in which the text is written. If uncertain, return null
- `gender`: The gender of the person. If uncertain, return null

Ensure the following:
- Only the specified information is included in the output.
- The output is a single JSON object.
- The JSON response starts with `{{` and ends with `}}`.
- No additional comments, notes, or extra information are included.
- Each document will have only one `document_number` key.

Output only the JSON data as described above.

Document:
{formatted_result}
""",
                    }
                ],
                model="llama-3.1-70b-versatile",
            )

            # Extract and return the processed data
            end_time = time.time()

            print(chat_completion.choices[0].message.content)

            return Response(
                {
                    "execution_time": end_time - start_time,
                    "ocr_result": json.loads(
                        "{"
                        + chat_completion.choices[0]
                        .message.content.strip()
                        .split("{")[1]
                        .split("}")[0]
                        + "}"
                    ),
                }
            )
        except Exception as e:
            return Response(
                {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
