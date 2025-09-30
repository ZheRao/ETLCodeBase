from ETLCodeBase import JobClasses 
import argparse


def main(args):
    #Extract and Transform
    if args.liveFX.lower() == "true":
        use_live_fx = True 
    else:
        use_live_fx = False
    QBOjob = JobClasses.QBOETL(use_live_fx=use_live_fx)
    if args.runQBO.lower() == "true":
        QBOjob.run(QBO_light=True, extract=True)
    else:
        if args.PLonly.lower() == "true":
            QBOjob.run(QBO_light=True, extract=True, PL_only=True)
        elif args.APonly.lower() == "true":
            QBOjob.run(QBO_light=True, extract=True, AP_only=True)
    QBOjob.run(QBO_light=True, extract=True)

    if args.runtsheet.lower() in ["normal", "full"]:
        QBOTimeJob = JobClasses.QBOTimeETL()
        if args.runtsheet.lower() == "full":
            QBOTimeJob.run(force_run=True)
        else:
            QBOTimeJob.run(force_run=False)

    # final transform
    # projects = JobClasses.Projects()
    # projects.run()

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        prog="ETL",
        description = "ETL Pipeline"
    )
    p.add_argument(
        "--runQBO",
        required= False,
        default = "true"
    )
    p.add_argument(
        "--PLonly",
        required = False,
        default = "false"
    )
    p.add_argument(
        "--APonly",
        required=False,
        default="false"
    )
    p.add_argument(
        "--runtsheet",
        required = False,
        choices = ["full", "normal", "none"],
        default="normal"
    )
    p.add_arguemtn(
        "--liveFX",
        required = False,
        default="true"
    )
    args = p.parse_args()
    main(args)

