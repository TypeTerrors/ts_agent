# syntax=docker/dockerfile:1

FROM node:20-alpine AS builder
WORKDIR /app
ARG VITE_WS_URL
ENV VITE_WS_URL=${VITE_WS_URL}
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
