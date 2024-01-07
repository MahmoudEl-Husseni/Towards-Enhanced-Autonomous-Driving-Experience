#ifndef SERIALMANAGER_H
#define SERIALMANAGER_H

#include <QObject>
#include <QSerialPort>
#include <QSerialPortInfo>

class SerialManager : public QObject
{
    Q_OBJECT
public:
    explicit SerialManager(QObject *parent = nullptr);
    ~SerialManager();

public slots:
    void readSerial();

signals:
    void temperatureChanged(const QString &newTemp);
    void speedChanged(double newSpeed);
    void doorStatusChanged(double newDoorStatus);
    void selectedGearChanged(QString selectedGear);

private:
    QSerialPort m_serial;
    bool m_connectStatus;
    static const quint16 m_serial_uno_vendor_id = 9025;
    static const quint16 m_serial_uno_product_id = 67;
    QByteArray serialData;
    QString serialBuffer;
    QString parsed_data;
    double temperature_value;
    double speed_value;
    double door_status;
    QString selected_Gear;
};

#endif // SERIALMANAGER_H
