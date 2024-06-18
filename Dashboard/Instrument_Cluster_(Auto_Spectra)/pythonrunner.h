// pythonrunner.h
#ifndef PYTHONRUNNER_H
#define PYTHONRUNNER_H

#include <QObject>
#include <QString>
#include <QProcess>

class PythonRunner : public QObject
{
    Q_OBJECT

public:
    explicit PythonRunner(QObject *parent = nullptr);

public slots:
    void runPythonScript();

signals:
    void pythonScriptOutput(const QString output);
    void pythonScriptError(const QString error);

private slots:
    void readStandardOutput();  // Corrected slot name
    void readStandardError();   // Added slot for standard error

    void processFinished(int exitCode, QProcess::ExitStatus exitStatus);

private:
    QProcess process;  // Declare QProcess as a member variable
};

#endif // PYTHONRUNNER_H
