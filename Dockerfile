FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /usr/src/app

# Install packages first so this layer is cached when only app code changes
COPY requirements.txt /
RUN pip install -r /requirements.txt

# Includes app/assets (dark-theme CSS and logo)
COPY app /usr/src/app/

RUN useradd -m appUser
USER appUser

EXPOSE 8050

CMD ["gunicorn", "--bind", "0.0.0.0:8050", "Nyquist_explorer:server"]