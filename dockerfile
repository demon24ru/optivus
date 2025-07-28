# compiler
FROM python:3.12-slim-bookworm AS stage-1
ARG TARGETARCH

ARG FISH_MODEL=fish-speech-1.5
ARG MINICPM_MODEL=MiniCPM-V-2_6-int4
ARG WHISPER_MODEL=whisper-base
ARG HF_ENDPOINT=https://huggingface.co

WORKDIR /opt/app

RUN set -ex \
    && pip install huggingface_hub \
    && apt-get -y install --no-install-recommends wget \
    && HF_ENDPOINT=${HF_ENDPOINT} huggingface-cli download --resume-download fishaudio/${FISH_MODEL} --local-dir checkpoints/${FISH_MODEL} \
    && huggingface-cli download --resume-download openbmb/${MINICPM_MODEL} --local-dir checkpoints/${MINICPM_MODEL} \
    && huggingface-cli download --resume-download openai/${WHISPER_MODEL} --local-dir checkpoints/${WHISPER_MODEL} \
    && wget -P .checkpoints https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth

# worker
FROM python:3.12-slim-bookworm
ARG TARGETARCH

ARG DEPENDENCIES="  \
    ca-certificates \
    libsox-dev \
    build-essential \
    cmake \
    libasound-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -ex \
    && rm -f /etc/apt/apt.conf.d/docker-clean \
    && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' >/etc/apt/apt.conf.d/keep-cache \
    && apt-get update \
    && apt-get -y install --no-install-recommends ${DEPENDENCIES} \
    && echo "no" | dpkg-reconfigure dash

WORKDIR /opt/app

COPY . .

RUN --mount=type=cache,target=/root/.cache,sharing=locked \
    set -ex \
    && pip install -e .[stable] \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install ./ImageBind

COPY --from=stage-1 /opt/app/checkpoints /opt/app/LLM
COPY --from=stage-1 /opt/app/.checkpoints /opt/app/.checkpoints

CMD ["./entrypoint.sh"]
