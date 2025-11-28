FROM python:3.11

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Install Chromium + Playwright runtime
RUN playwright install --with-deps chromium

COPY --chown=user . /app

ENV PORT=7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
