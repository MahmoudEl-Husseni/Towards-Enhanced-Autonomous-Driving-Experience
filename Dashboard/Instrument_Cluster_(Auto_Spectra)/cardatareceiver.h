#ifndef CARDATARECEIVER_H
#define CARDATARECEIVER_H

#include <QObject>
#include <QUdpSocket>

class CarDataReceiver : public QObject {
    Q_OBJECT
public:
    explicit CarDataReceiver(QObject *parent = nullptr);

signals:
    void carlaJsonDataParsed(int speed,QString alart,int sign, QString autoGear,int leftSignal,int rightSignal,int warning,int handBrake, double brake);

private slots:
    void processPendingDatagrams();

private:
    QUdpSocket udpSocket;
};

#endif // CARDATARECEIVER_H
