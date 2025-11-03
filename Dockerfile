# syntax=docker/dockerfile:1

FROM node:22.13.1-slim AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY tsconfig.json ./
COPY src ./src

RUN npm run build

FROM node:22.13.1-slim AS runner
ENV NODE_ENV=production
WORKDIR /app

COPY --from=builder /app/package.json ./
COPY --from=builder /app/package-lock.json ./
RUN npm ci --omit=dev

COPY --from=builder /app/dist ./dist
COPY README.md ./README.md

RUN mkdir -p /app/logs /app/models

ENV LOG_FILE=logs/trading.log \
    MODEL_STORE_PATH=models

CMD ["node", "dist/index.js"]
