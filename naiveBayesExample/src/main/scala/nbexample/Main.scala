package nbexample


object Main {

  def main(args: Array[String]): Unit = {
    if (args.length < 1) {
      println("Usage: countryserver.sh <port>")
      System.exit(1)
    }

    val nbServer = new TomcatServer()
    val port = args(0).toInt match {
      case xx if xx > 0 => xx
      case _ => 8079
    }
    println("Starting server on port: "+port)
    nbServer.startServer(port)

  }

}
