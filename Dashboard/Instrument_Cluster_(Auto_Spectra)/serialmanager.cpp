#include <QJsonDocument>
#include <QJsonObject>
#include <QDebug>
#include "serialmanager.h"
using namespace std;
SerialManager::SerialManager(QObject *parent)
    : QObject(parent)
{
    udpSocket.bind(QHostAddress::Any, 12345); // Use appropriate port number
    bool m_serial_is_available = true;
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
        // m_serial.setPortName("/dev/ttyAMA0");
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
    udpSocket.close();
}

void SerialManager::readSerial()
{
    static QByteArray receivedData;

    while (m_serial.canReadLine())
    {
        QByteArray data = m_serial.readLine();
        receivedData.append(data);
        if (data.endsWith('\n'))
        {
            // Parse the received JSON object
            QJsonParseError jsonError;
            QJsonDocument jsonDoc = QJsonDocument::fromJson(receivedData, &jsonError);
            if (jsonError.error != QJsonParseError::NoError)
            {
                qDebug() << "Failed to parse JSON data:" << jsonError.errorString();
            }
            else if (jsonDoc.isObject())
            {
                QJsonObject jsonObj = jsonDoc.object();
                int temperature = jsonObj["temperature"].toInt();
                int humidity = jsonObj["humidity"].toInt();
                int door = jsonObj["door"].toInt();
                QString gear = jsonObj["gear"].toString();
                sendGearOverUDP(gear);
                emit jsonDataParsed(temperature, humidity, door, gear);
//                qDebug() << "Temperature:" << temperature;
//                qDebug() << "Humidity:" << humidity;
//                qDebug() << "Door:" << door;
                // qDebug() << "Gear:" << gear;
            }

            // Reset receivedData for the next JSON object
            receivedData.clear();
        }
    }
}

void SerialManager::sendGearOverUDP(const QString &gear)
{
    // Create a JSON object to hold the gear information
    QJsonObject json;
    json["gear"] = gear;

    // Create a JSON document from the JSON object
    QJsonDocument jsonDoc(json);

    // Convert the JSON document to bytes
    QByteArray jsonData = jsonDoc.toJson();

    // Send the JSON data over UDP
    udpSocket.writeDatagram(jsonData, QHostAddress("192.168.1.5"), 12345); // Replace with the IP and port of the Python PC
}

