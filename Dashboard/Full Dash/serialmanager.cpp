#include <QDebug>
#include "serialmanager.h"
using namespace std;
SerialManager::SerialManager(QObject *parent)
    : QObject(parent)
{
    serialBuffer = "";
    parsed_data = "";
    temperature_value = 0.0;

    bool m_serial_is_available = false;
    QString m_serial_uno_port_name;
    //  For each available serial port
    foreach(const QSerialPortInfo &serialPortInfo, QSerialPortInfo::availablePorts()){
        //  check if the serialport has both a product identifier and a vendor identifier
        if(serialPortInfo.hasProductIdentifier() && serialPortInfo.hasVendorIdentifier()){
            //  check if the product ID and the vendor ID match those of the m_serial uno
            if((serialPortInfo.productIdentifier() == m_serial_uno_product_id)
                && (serialPortInfo.vendorIdentifier() == m_serial_uno_vendor_id)){
                m_serial_is_available = true; //    m_serial uno is available on this port
                m_serial_uno_port_name = serialPortInfo.portName();
            }
        }
    }

    if(m_serial_is_available){
        qDebug() << "Found the m_serial port...\n";
        m_serial.setPortName(m_serial_uno_port_name);
        m_serial.open(QSerialPort::ReadOnly);
        m_serial.setBaudRate(QSerialPort::Baud9600);
        m_serial.setDataBits(QSerialPort::Data8);
        m_serial.setFlowControl(QSerialPort::NoFlowControl);
        m_serial.setParity(QSerialPort::NoParity);
        m_serial.setStopBits(QSerialPort::OneStop);
        connect(&m_serial, SIGNAL(readyRead()), this, SLOT(readSerial()));
    }else{
        qDebug() << "Couldn't find the correct port for the m_serial.\n";
    }

}

SerialManager::~SerialManager()
{
    // If serial port is open close it
    if( m_serial.isOpen() )
    {
        m_serial.close();
    }
}

void SerialManager::readSerial()
{
    // Read all available data from the serial port
    serialData = m_serial.readAll();
    serialBuffer += QString::fromStdString(serialData.toStdString());

    // Split the buffer by ',' to get individual readings
    QStringList buffer_split = serialBuffer.split(',');

    // Check if we have at least 5 elements in buffer_split
    if (buffer_split.length() >= 5) {
        // Extract values
        temperature_value = buffer_split[0].toInt();
        door_status = buffer_split[1].toInt();
        speed_value = buffer_split[2].toInt();
        selected_Gear = buffer_split[3];
        // Clear the buffer
        serialBuffer.clear();

        // Emit signals for each variable
        emit temperatureChanged(QString::number(temperature_value));
        emit speedChanged(speed_value);
        emit doorStatusChanged(door_status);
        emit selectedGearChanged(selected_Gear);


        qDebug() << "Temperature: " << temperature_value;
        qDebug() << "Speed: " << speed_value;
        qDebug() << "Door Status: " << door_status;
        qDebug() << "Selected Gear: " << selected_Gear;

    }
}

