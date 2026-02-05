#!/bin/bash

# MCP Memory Service Control Script
SERVICE_NAME="mcp-memory"

case "$1" in
    start)
        echo "Starting MCP Memory Service..."
        sudo systemctl start $SERVICE_NAME
        sleep 2
        sudo systemctl status $SERVICE_NAME --no-pager
        ;;
    stop)
        echo "Stopping MCP Memory Service..."
        sudo systemctl stop $SERVICE_NAME
        sudo systemctl status $SERVICE_NAME --no-pager
        ;;
    restart)
        echo "Restarting MCP Memory Service..."
        sudo systemctl restart $SERVICE_NAME
        sleep 2
        sudo systemctl status $SERVICE_NAME --no-pager
        ;;
    status)
        sudo systemctl status $SERVICE_NAME --no-pager
        ;;
    logs)
        echo "Showing recent logs (Ctrl+C to exit)..."
        sudo journalctl -u $SERVICE_NAME -f
        ;;
    health)
        echo "Checking service health..."
        curl -k -s https://localhost:8000/api/health | jq '.' 2>/dev/null || curl -k -s https://localhost:8000/api/health
        ;;
    enable)
        echo "Enabling service for startup..."
        sudo systemctl enable $SERVICE_NAME
        echo "Service will start automatically on boot"
        ;;
    disable)
        echo "Disabling service from startup..."
        sudo systemctl disable $SERVICE_NAME
        echo "Service will not start automatically on boot"
        ;;
    install)
        echo "Installing service..."
        ./install_service.sh
        ;;
    uninstall)
        echo "Uninstalling service..."
        sudo systemctl stop $SERVICE_NAME 2>/dev/null
        sudo systemctl disable $SERVICE_NAME 2>/dev/null
        sudo rm -f /etc/systemd/system/$SERVICE_NAME.service
        sudo systemctl daemon-reload
        echo "Service uninstalled"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|health|enable|disable|install|uninstall}"
        echo ""
        echo "Commands:"
        echo "  start     - Start the service"
        echo "  stop      - Stop the service"
        echo "  restart   - Restart the service"
        echo "  status    - Show service status"
        echo "  logs      - Show live service logs"
        echo "  health    - Check API health endpoint"
        echo "  enable    - Enable service for startup"
        echo "  disable   - Disable service from startup"
        echo "  install   - Install the systemd service"
        echo "  uninstall - Remove the systemd service"
        exit 1
        ;;
esac
