/// This is an example of Ray C++ application. Please visit
/// `https://docs.ray.io/en/master/index.html` for more details.

/// including the `<ray/api.h>` header
#include <ray/api.h>

/// common function
int Plus(int x, int y) { return x + y; }
/// Declare remote function
RAY_REMOTE(Plus);

/// class
class Counter {
 public:
  int count;

  Counter(int init = 0) { count = init; }

  int Increment() {
    count += 1;
    return count;
  }

  /// static factory method
  static Counter *FactoryCreate(int init) { return new Counter(init); }

  /// non static function
  int Add(int x) {
    count += x;
    return count;
  }
};
/// Declare remote function
RAY_REMOTE(Counter::FactoryCreate, &Counter::Add);

#if 1
// Factory function of Counter class.
static Counter *CreateCounter() {
    return new Counter();
};
RAY_REMOTE(&Counter::Increment, CreateCounter);

#endif

int main(int argc, char **argv) {
  /// initialization
  ray::Init();

  /// put and get object
  auto object = ray::Put(100);
  auto put_get_result = *(ray::Get(object));
  std::cout << "put_get_result = " << put_get_result << std::endl;

  /// common task
  auto task_object = ray::Task(Plus).Remote(1, 2);
  int task_result = *(ray::Get(task_object));
  std::cout << "task_result = " << task_result << std::endl;

  /// actor
  ray::ActorHandle<Counter> actor = ray::Actor(Counter::FactoryCreate).Remote(0);
  /// actor task
  auto actor_object = actor.Task(&Counter::Add).Remote(3);
  int actor_task_result = *(ray::Get(actor_object));
  std::cout << "actor_task_result = " << actor_task_result << std::endl;
  /// actor task with reference argument
  auto actor_object2 = actor.Task(&Counter::Add).Remote(task_object);
  int actor_task_result2 = *(ray::Get(actor_object2));
  std::cout << "actor_task_result2 = " << actor_task_result2 << std::endl;

  // Create an actor from this class.
  // `ray::Actor` takes a factory method that can produce
  // a `Counter` object. Here, we pass `Counter`'s factory function
  // as the argument.
  auto counter = ray::Actor(CreateCounter).Remote();

  // Call the actor.
  auto object_ref = counter.Task(&Counter::Increment).Remote();
  assert(*object_ref.Get() == 1);

#if 1
  // Create ten Counter actors.
  std::vector<ray::ActorHandle<Counter>> counters;
  for (int i = 0; i < 10; i++) {
    counters.emplace_back(ray::Actor(CreateCounter).Remote());
  }

  // Increment each Counter once and get the results. These tasks all happen in
  // parallel.
  std::vector<ray::ObjectRef<int>> object_refs;
  for (ray::ActorHandle<Counter> counter_actor : counters) {
    object_refs.emplace_back(counter_actor.Task(&Counter::Increment).Remote());
  }
  // prints 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
  auto results = ray::Get(object_refs);
  for (const auto &result : results) {
    std::cout << *result;
  }

  // Increment the first Counter five times. These tasks are executed serially
  // and share state.
  object_refs.clear();
  for (int i = 0; i < 5; i++) {
    object_refs.emplace_back(counters[0].Task(&Counter::Increment).Remote());
  }
  // prints 2, 3, 4, 5, 6
  results = ray::Get(object_refs);
  for (const auto &result : results) {
    std::cout << *result;
  }
#endif

  /// shutdown
  ray::Shutdown();
  return 0;
}
