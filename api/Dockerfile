# syntax=docker/dockerfile:1

FROM golang:1.23-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN go build -o wsapi ./cmd/wsapi

FROM alpine:3.20
WORKDIR /app
COPY --from=builder /app/wsapi ./wsapi
EXPOSE 8080
ENV API_PORT=8080
CMD ["/app/wsapi"]
