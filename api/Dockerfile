# syntax=docker/dockerfile:1

FROM golang:1.25.2-alpine AS builder

WORKDIR /app
ENV GOTOOLCHAIN=local
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download

COPY . .
RUN --mount=type=cache,target=/go/pkg/mod --mount=type=cache,target=/root/.cache/go-build go build -o wsapi ./cmd/wsapi

FROM alpine:3.20
WORKDIR /app
COPY --from=builder /app/wsapi ./wsapi
EXPOSE 8080
ENV API_PORT=8080
CMD ["/app/wsapi"]
