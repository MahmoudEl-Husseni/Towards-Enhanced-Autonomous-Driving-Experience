#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include "serialmanager.h"
#include "pythonrunner.h"
#include "cardatareceiver.h"
#include "SpotifyReceiver.h"



int main(int argc, char *argv[])
{
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif

    QGuiApplication app(argc, argv);
    qmlRegisterType<SerialManager>("com.company.serialmanager", 1, 0, "SerialManager" );
    qmlRegisterType<PythonRunner>("MyPythonScript", 1, 0, "PythonRunner");
    qmlRegisterType<CarDataReceiver>("com.company.cardatareceiver", 1, 0, "CarDataReceiver" );
    qmlRegisterType<SpotifyReceiver>("spotifyreceiver", 1, 0, "Spotifyreceiver" );

    QQmlApplicationEngine engine;

    SerialManager serialManager;
    engine.rootContext()->setContextProperty("serialManager", &serialManager);

    PythonRunner pythonRunner;
    engine.rootContext()->setContextProperty("pythonRunner", &pythonRunner);

    CarDataReceiver carDataReceiver;
    engine.rootContext()->setContextProperty("carDataReceiver", &carDataReceiver);

    SpotifyReceiver SpotifyReceiver;
    engine.rootContext()->setContextProperty("SpotifyReceiver", &SpotifyReceiver);

    const QUrl url(QStringLiteral("qrc:/main.qml"));
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
                     &app, [url](QObject *obj, const QUrl &objUrl) {
        if (!obj && url == objUrl){
            QCoreApplication::exit(-1);
        }
    }, Qt::QueuedConnection);

    engine.load(url);


    // Run the Python script when the QML component is completed
    pythonRunner.runPythonScript();

    return app.exec();
}
