#!/bin/bash

echo "EGNNN-GROA Cloud IDS Deployment Script"
echo "======================================"
echo ""

# Update and upgrade system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install required packages
echo "Installing Python, pip, nginx..."
sudo apt install python3 python3-pip python3-venv nginx -y

# Navigate to project directory
cd ~/ids_project

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv ids_env

# Activate virtual environment
echo "Activating virtual environment..."
source ids_env/bin/activate

# Install Python packages
echo "Installing Python dependencies..."
pip install -r requirements.txt
pip install flask flask-socketio flask-cors gunicorn eventlet

# Create systemd service
echo "Creating systemd service..."
sudo tee /etc/systemd/system/ids.service > /dev/null <<EOF
[Unit]
Description=EGNNN IDS API Server
After=network.target

[Service]
WorkingDirectory=/home/ubuntu/ids_project
ExecStart=/home/ubuntu/ids_project/ids_env/bin/gunicorn --worker-class eventlet -w 1 cloud_ids_api:app --bind 0.0.0.0:5000
Restart=always
User=ubuntu

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
echo "Enabling and starting IDS service..."
sudo systemctl enable ids
sudo systemctl start ids

# Configure nginx
echo "Configuring nginx reverse proxy..."
sudo tee /etc/nginx/sites-available/ids > /dev/null <<EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /socket.io {
        proxy_pass http://localhost:5000/socket.io;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable nginx site
sudo ln -sf /etc/nginx/sites-available/ids /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Get public IP
PUBLIC_IP=$(curl -s http://checkip.amazonaws.com)

echo ""
echo "Deployment completed successfully!"
echo "=================================="
echo "IDS API Server: http://$PUBLIC_IP"
echo "Dashboard: http://$PUBLIC_IP/"
echo ""
echo "Service status:"
sudo systemctl status ids --no-pager
echo ""
echo "To check logs:"
echo "sudo journalctl -u ids -f"
echo ""
echo "To restart service:"
echo "sudo systemctl restart ids"