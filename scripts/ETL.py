from ETLCodeBase import JobClasses 


def main():
    # Extract and Transform
    QBOjob = JobClasses.QBOETL()
    QBOjob.run(QBO_light=True, extract=True)

    QBOTimeJob = JobClasses.QBOTimeETL()
    QBOTimeJob.run(force_run=False)

    # final transform
    # projects = JobClasses.Projects()
    # projects.run()

if __name__ == "__main__":
    main()

