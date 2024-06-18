#ifndef SERIALMANAGER_H
#define SERIALMANAGER_H

#include <QObject>
#include <QSerialPort>
#include <QSerialPortInfo>
#include <QUdpSocket>
#include <QJsonDocument>
#include <QJsonObject>

class SerialManager : public QObject
{
    Q_OBJECT
public:
    explicit SerialManager(QObject *parent = nullptr);
    ~SerialManager();

public slots:
    void readSerial();
    void sendGearOverUDP(const QString &gear);

signals:
    void jsonDataParsed(int temperature, int humidity, int door, QString gear);

private:
    QSerialPort m_serial;
    bool m_connectStatus;
    static const quint16 m_serial_uno_vendor_id = 0x0403;
    static const quint16 m_serial_uno_product_id = 0x6001;
    QByteArray serialData;
    QString serialBuffer;
    QString parsed_data;

    QUdpSocket udpSocket;
    static const quint16 udpPort = 1234;
};

#endif // SERIALMANAGER_H
