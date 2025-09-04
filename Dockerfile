FROM mcr.microsoft.com/playwright:v1.40.0-focal

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .

# Install browsers (if not using base image)
RUN npx playwright install --with-deps

CMD ["npm", "test"]
