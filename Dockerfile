FROM node:18

# Set direktori kerja
WORKDIR /usr/src/app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies only for production
RUN npm install --only=production

# Set environment variables
ENV NODE_ENV production

# Expose the port defined by Cloud Run (use $PORT)
EXPOSE $PORT

# Copy all source files
COPY . .

# Use the PORT environment variable in the start script
CMD [ "node", "index.js" ]