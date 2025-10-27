from ETLCodeBase import JobClasses 
import argparse


def main(args):

    if args.HPonly.lower() == "false": # don't execute these if HPonly == True
        #Extract and Transform
        if args.liveFX.lower() == "true" and args.APonly.lower() == "false":        # no live FX rate for AP report
            use_live_fx = True 
        else:
            use_live_fx = False
        
        QBOjob = JobClasses.QBOETL(use_live_fx=use_live_fx)
        if args.extract.lower() == "true":
            extract = True 
        else:
            extract = False
        

        if args.PLonly.lower() == "true":
            QBOjob.run(QBO_light=True, extract=extract, PL_only=True)
            return
        elif args.APonly.lower() == "true":
            QBOjob.run(QBO_light=True, extract=extract, AP_only=True)
            return
        elif args.runQBO.lower() == "true":
            QBOjob.run(QBO_light=True, extract=extract)

        if args.runtsheet.lower() in ["normal", "full"]:
            QBOTimeJob = JobClasses.QBOTimeETL()
            if args.runtsheet.lower() == "full":
                QBOTimeJob.run(force_run=True)
            else:
                QBOTimeJob.run(force_run=False)
    
    hpjob = JobClasses.HPETL()
    hpjob.run()

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
    p.add_argument(
        "--liveFX",
        required = False,
        default="true"
    )
    p.add_argument(
        "--HPonly",
        required=False,
        default="false"
    )
    p.add_argument(
        "--extract",
        required=False,
        default="true"
    )
    args = p.parse_args()
    main(args)

