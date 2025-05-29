using FAIR.API.Hubs;
using FAIR.Application.DependencyInjenction;
using FAIR.Infrastructure.DependencyInjection;
using Serilog;
var builder = WebApplication.CreateBuilder(args);
Log.Logger = new LoggerConfiguration()
    .Enrich.FromLogContext()
    .WriteTo.Console()
    .WriteTo.File("log/log.txt", rollingInterval: RollingInterval.Day)
    .CreateLogger();
builder.Host.UseSerilog();
Log.Logger.Information("Application is Building.........");
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddInfrastructureService(builder.Configuration);
builder.Services.AddApplicationService();
builder.Services.AddSignalR();
builder.Services.AddCors(builder => {
    builder.AddDefaultPolicy(options =>
    {
        options.AllowAnyHeader()
        .AllowAnyMethod()
        .WithOrigins()
        .AllowCredentials();
    });
});

try
{
    var app = builder.Build();
    app.UseSerilogRequestLogging();
    // Configure the HTTP request pipeline.
  //  if (app.Environment.IsDevelopment())
   // {
        app.UseSwagger();
        app.UseSwaggerUI();
   // }
    app.UseInfrastructureService();
    app.UseHttpsRedirection();

    app.UseAuthorization();

    app.MapControllers();
    app.MapHub<ChatHub>("/chat");
    Log.Logger.Information("Application is running........");
    app.Run();
}
catch(Exception ex )
{
    Log.Logger.Error(ex, "Application failed to Start....");
}
finally
{
    Log.CloseAndFlush();
}

