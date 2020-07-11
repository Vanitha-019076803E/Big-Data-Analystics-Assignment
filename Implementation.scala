
//Question 1
// Reduce some of the debugging output of Spark
import org.apache.log4j.Logger
import org.apache.log4j.Level
Logger.getLogger("org").setLevel(Level.ERROR)
Logger.getLogger("akka").setLevel(Level.ERROR)

// Import the basic recommender libraries from Spark's MLlib package
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

val rawArtistAlias = sc.textFile("/home/users/vthamilselvan/vanitha/audioscrobbler/artist_alias.txt")
val rawArtistData = sc.textFile("/home/users/vthamilselvan/vanitha/audioscrobbler/artist_data.txt")
val rawUserArtistData = sc.textFile("/home/users/vthamilselvan/vanitha/audioscrobbler/user_artist_data.txt")

val artistByID = rawArtistData.flatMap {
  line => val (id, name) = line.span(_ != '\t')
  if (name.isEmpty) {
    None
  } else {
    try {
      Some((id.toInt, name.trim)) }
    catch { case e: NumberFormatException => None }
  }
}

val artistAlias = rawArtistAlias.flatMap { line =>
  val tokens = line.split('\t')
  if (tokens(0).isEmpty) {
    None
  } else {
    Some((tokens(0).toInt, tokens(1).toInt))
  }
}.collectAsMap()


// Broadcasting the local aliases map which is going to be part of the closure of our training function
val bArtistAlias = sc.broadcast(artistAlias)

// Preparing and caching the entire data
val data = rawUserArtistData.map {
  line =>
    val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
    val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
    Rating(userID, finalArtistID, count)
    }.cache()


val fractions = data.map(x => x.user).distinct.map(x => (x,0.9)).collectAsMap()
val byKey = data.keyBy(x => x.user)

//90% split into train data
val trainData = byKey.sampleByKeyExact(false, fractions).map(_._2).cache()

// 10% split into test data
val testData = data.subtract(trainData).cache()

trainData.count()
testData.count()

// And training the recommender model using Spark's ALS algorithm
val model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)

//Taking random users from testData
val someUsers = testData.map(x => x.user).distinct().takeSample(false,500)


// Using recommendProducts function  to find the top 100 recommendations of 500 selected users
val recommendations = someUsers.flatMap(userID => model.recommendProducts(userID, 100))

// auc for recommendations of all the users
//(y^,y) pairs given as the input of the BinaryClassificationMetrics
val aucList = (for(i <- 0 to (someUsers.length-1))
yield
{
// actual artists for the corresponding users taken from test Data
val actualArtistsForUser = testData.filter(x => x.user == someUsers(i)).collect.map(x => x.product)
sc.parallelize(recommendations.filter(x => x.user == someUsers(i)).map(x =>
if(actualArtistsForUser.contains(x.product))
{
 (x.rating, 1.toDouble)
} else {
 (x.rating, 0.toDouble)
}))}).map(x => new BinaryClassificationMetrics(x).areaUnderROC) // finding auc using BinaryClassificationMetrics API

//average AUC
val avgAUC = aucList.sum/aucList.length

//Compare the results to a baseline model which simply recommends the same most popular artists to each user
def predictMostPopular(user: Int, numArtists: Int) = {
   trainData.map(r => (r.product, r.rating)).reduceByKey(_ + _).collect().sortBy(-_._2).toList.take(numArtists).map{case (artist, rating) =>
   Rating(user, artist, rating)}
}

// Using predictMostPopular function  to find the top 100 recommendations of 500 selected users
val recommendations1 = someUsers.flatMap(userID => predictMostPopular(userID, 100))

// auc for recommendations of all the users
//(y^,y) pairs given as the input of the BinaryClassificationMetrics
val aucList1 = (for(i <- 0 to (someUsers.length-1))
yield
{
val actualArtistsForUser = data.filter(x => x.user == someUsers(i)).collect.map(x => x.product)
sc.parallelize(recommendations1.filter(x => x.user == someUsers(i)).map(x =>
if(actualArtistsForUser.contains(x.product))
{
 (x.rating, 1.toDouble)
} else {
 (x.rating, 0.toDouble)
}))}).map(x => new BinaryClassificationMetrics(x).areaUnderROC).toList

// Average of AUC
val avgAUC1 = aucList1.sum/aucList1.length

********************************************************************************************************************************************************
// Question 3.2

// Reduce some of the debugging output of Spark
import org.apache.log4j.Logger
import org.apache.log4j.Level
Logger.getLogger("org").setLevel(Level.ERROR)
Logger.getLogger("akka").setLevel(Level.ERROR)

// Import the basic recommender libraries from Spark's MLlib package
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

val rawArtistAlias = sc.textFile("/home/users/vthamilselvan/vanitha/audioscrobbler/artist_alias.txt")
val rawArtistData = sc.textFile("/home/users/vthamilselvan/vanitha/audioscrobbler/artist_data.txt")
val rawUserArtistData = sc.textFile("/home/users/vthamilselvan/vanitha/audioscrobbler/user_artist_data.txt")

val artistByID = rawArtistData.flatMap {
  line => val (id, name) = line.span(_ != '\t')
  if (name.isEmpty) {
    None
  } else {
    try {
      Some((id.toInt, name.trim)) }
    catch { case e: NumberFormatException => None }
  }
}

val artistAlias = rawArtistAlias.flatMap { line =>
  val tokens = line.split('\t')
  if (tokens(0).isEmpty) {
    None
  } else {
    Some((tokens(0).toInt, tokens(1).toInt))
  }
}.collectAsMap()


// Broadcasting the local aliases map which is going to be part of the closure of our training function
val bArtistAlias = sc.broadcast(artistAlias)

// Preparing and caching the entire data
val data = rawUserArtistData.map {
  line =>
    val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
    val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
    Rating(userID, finalArtistID, count)
    }.cache()


val fractions = data.map(x => x.user).distinct.map(x => (x,0.9)).collectAsMap()
val byKey = data.keyBy(x => x.user)

//90% split into train data
val trainData = byKey.sampleByKeyExact(false, fractions).map(_._2).cache()

// 10% split into test data
val testData = data.subtract(trainData).cache()

trainData.count()
testData.count()

//Taking random users from testData
val someUsers = testData.map(x => x.user).distinct().takeSample(false,500)


//AUC measure of your model over the 10% split
val start = System.nanoTime();
val evaluations = for(rank <- Array(10, 25);
lambda <- Array(1.0);
alpha <- Array(1.0))
yield {
val model = ALS.trainImplicit(trainData, rank, 10, lambda, alpha) //taking test data to train the model
val recommendations = someUsers.flatMap(userID => model.recommendProducts(userID, 100))
val aucList = (for(i <- 0 to (someUsers.length-1)) // auc for recommendations of all the users
yield
{
val actualArtistsForUser = testData.filter(x => x.user == someUsers(i)).collect.map(x => x.product)
sc.parallelize(recommendations.filter(x => x.user == someUsers(i)).map(x =>
if(actualArtistsForUser.contains(x.product))
{
 (x.rating, 1.toDouble)
} else {
 (x.rating, 0.toDouble)
}))}).map(x => new BinaryClassificationMetrics(x).areaUnderROC)
val avgAUC = aucList.sum/aucList.length // average of AUC
((rank, lambda, alpha),avgAUC)
}
val end = System.nanoTime();

println(" Overall Duration::::::::::::"+ (end-start)/1e9d);

evaluations.sortBy(_._2).reverse.foreach(println)


********************************************************************************************************************************************************



//Question 3.3

// Reduce some of the debugging output of Spark
import org.apache.log4j.Logger
import org.apache.log4j.Level
Logger.getLogger("org").setLevel(Level.ERROR)
Logger.getLogger("akka").setLevel(Level.ERROR)
import org.apache.spark.mllib.util.MLUtils
// Import the basic recommender libraries from Spark's MLlib package
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

val rawArtistAlias = sc.textFile("/home/users/vthamilselvan/vanitha/audioscrobbler/artist_alias.txt")
val rawArtistData = sc.textFile("/home/users/vthamilselvan/vanitha/audioscrobbler/artist_data.txt")
val rawUserArtistData = sc.textFile("/home/users/vthamilselvan/vanitha/audioscrobbler/user_artist_data.txt")

val artistByID = rawArtistData.flatMap {
  line => val (id, name) = line.span(_ != '\t')
  if (name.isEmpty) {
    None
  } else {
    try {
      Some((id.toInt, name.trim)) }
    catch { case e: NumberFormatException => None }
  }
}

val artistAlias = rawArtistAlias.flatMap { line =>
  val tokens = line.split('\t')
  if (tokens(0).isEmpty) {
    None
  } else {
    Some((tokens(0).toInt, tokens(1).toInt))
  }
}.collectAsMap()


// Broadcasting the local aliases map which is going to be part of the closure of our training function
val bArtistAlias = sc.broadcast(artistAlias)

// Preparing and caching the entire data
val data = rawUserArtistData.map {
  line =>
    val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
    val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
    Rating(userID, finalArtistID, count)
    }.cache()


val fractions = data.map(x => x.user).distinct.map(x => (x,0.9)).collectAsMap()
val byKey = data.keyBy(x => x.user)

//90% split into train data
val trainData = byKey.sampleByKeyExact(false, fractions).map(_._2).cache()

// 10% split into test data
val testData = data.subtract(trainData).cache()

trainData.count()
testData.count()

//Taking random users from testData
val someUsers = testData.map(x => x.user).distinct().takeSample(false,500)

val someUsers = testData.map(x => x.user).distinct().takeSample(false,2)

// Using MLUtils.kFold() API to implement the cross-validations with AUC

val start = System.nanoTime();
val kFoldData = MLUtils.kFold(trainData, 5, 0);
  val evaluations = for(fold <- kFoldData)
  yield
   {
    val trainFoldData = fold._1
    val testFoldData = fold._2
    val model = ALS.trainImplicit(trainFoldData, 5, 10, 1.0, 1.0)
    val recommendations = someUsers.flatMap(userID => model.recommendProducts(userID, 100))
    val aucList = (for(i <- 0 to (someUsers.length-1))
    yield
    {
      val actualArtistsForUser = testData.filter(x => x.user == someUsers(i)).collect.map(x => x.product)
      sc.parallelize(recommendations.filter(x => x.user == someUsers(i)).map(x =>
                      if(actualArtistsForUser.contains(x.product))
                      {
                        (x.rating, 1.toDouble)
                      } else {
                        (x.rating, 0.toDouble)
                      })
                    )
    }
    ).map(x => new BinaryClassificationMetrics(x).areaUnderROC)
    val auc = aucList.sum/aucList.length
    ((10, 1.0, 1.0),auc)
    }
    val end = System.nanoTime();
    println(" Overall Duration::::::::::::"+ (end-start)/1e9d);
    evaluations.sortBy(_._2).reverse.foreach(println)
