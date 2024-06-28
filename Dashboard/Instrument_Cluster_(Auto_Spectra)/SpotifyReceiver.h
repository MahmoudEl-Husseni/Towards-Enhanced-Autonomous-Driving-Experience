#ifndef SPOTIFYRECEIVER_H
#define SPOTIFYRECEIVER_H

#include <QObject>
#include <QUdpSocket>

class SpotifyReceiver : public QObject
{
    Q_OBJECT
public:
    explicit SpotifyReceiver(QObject *parent = nullptr);

signals:
    void spotifiyReceivedData(QString Track_Name,QString Artist_Name,QString Album_Name,QString Album_Img_URL);

public slots:
    void processPendingDatagrams();

private:
    QUdpSocket udpSocket;
};

#endif // SPOTIFYRECEIVER_H
