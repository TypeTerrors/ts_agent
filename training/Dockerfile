# syntax=docker/dockerfile:1

FROM node:22.13.1-slim AS builder

WORKDIR /app/training

COPY training/package.json training/package-lock.json ./
RUN npm ci

COPY training/tsconfig.json ./
COPY training/src ./src

RUN npm run build

FROM node:22.13.1-slim AS runner
ENV NODE_ENV=production
WORKDIR /app/training

COPY --from=builder /app/training/package.json ./
COPY --from=builder /app/training/package-lock.json ./
RUN npm ci --omit=dev

COPY --from=builder /app/training/dist ./dist
COPY README.md /app/README.md

RUN mkdir -p /app/logs /app/models

ENV LOG_FILE=/app/logs/trading.log \
    MODEL_STORE_PATH=/app/models

CMD ["node", "dist/index.js"]
